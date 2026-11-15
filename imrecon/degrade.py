"""Turn a 3 T scan into a 64 mT-like one in steps, so each step's share of the match can be measured.

    0 resampled    the 3 T volume moved onto the 64 mT grid (rigid registration), scaled to white matter
    1 resolution   before that, blurred to the 64 mT voxel: in-plane Gaussian of FWHM one voxel, and a box as long as
                   the slice is thick along the slice axis (a 2D slice profile, not a Gaussian)
    2 contrast     a piecewise-linear map through paired class means: four intensity classes found on the blurred 3 T,
                   each eroded to its pure core, and the mean there of both the 3 T and another 64 mT scan of the same
                   person (registered onto the grid); one map for the brain, one for the rest of the head; never the
                   scan the fake is compared with. Three other maps were tried and left out: a voxel-wise lookup
                   (median 64 mT value per 3 T intensity) is pulled toward the mean by partial volume and small
                   misregistration; quantile (histogram) matching copies the 64 mT histogram's noise-widened tails
                   onto a nearly noise-free image; and pairing class means by rank maps 3 T white matter onto 64 mT
                   grey matter, because 3 T has a dark class (vessels) that 64 mT does not
    2b physics     instead of an intensity map: the 3 T scan (1 mm, before any blur) split into tissue fractions
                   (air or bone, white matter, grey matter, CSF; linear mixing between neighbouring class means), each
                   given its 64 mT spin-echo signal PD (1 - exp(-TR/T1)) exp(-TE/T2) from the 50 mT table
                   (O'Reilly & Webb 2022) at the scan's TR and TE, then blurred and resampled. Partial volume now mixes
                   signals, as it does in the scanner, instead of 3 T intensities
    2c fitted      the same fractions, with each tissue's 64 mT signal fitted by least squares to another 64 mT scan of
                   the same person (the blurred fraction maps against it, in the brain)
    3 noise        on 2c: Gaussian, blurred in-plane by `smoothing_px`, SD = white matter / `snr` (the measured medians:
                   SNR 8.3 from rescans, 0.6 px); the air outside the head is set to 0, as Hyperfine's masking leaves it

The distribution fits found the noise near-Gaussian (Student-t, nu >= 10, worth 0.01 bits), so a Gaussian is used.
"""

import numpy as np
from scipy import ndimage as ndi

from . import noise as N

SNR = 8.3
SMOOTHING_PX = 0.6
STAGES = ("0 resampled", "1 + resolution", "2a + contrast: map", "2b + contrast: physics", "2c + contrast: fitted",
          "3 + noise")
TR_S, TE_S = 2.0, 0.161                              # Hyperfine T2w in ds006557
PD = {"wm": 0.685, "gm": 0.815, "csf": 0.975}        # mid-range proton densities, as lowfield-sim draws them
T1T2_50MT = {"wm": (0.275, 0.102), "gm": (0.327, 0.102), "csf": (3.695, 1.584)}   # O'Reilly & Webb 2022


def register_transform(fixed, moving):
    """The rigid transform taking `moving` onto `fixed` (SimpleITK images): two starting points (centre of mass,
    image centre) and two metrics (correlation, mutual information), keeping the one whose result correlates best
    with `fixed` inside its head."""
    import SimpleITK as sitk
    f = N.to_array(fixed)
    head = N.head_mask(f)
    best, best_r = None, -np.inf
    for init in (sitk.CenteredTransformInitializerFilter.MOMENTS, sitk.CenteredTransformInitializerFilter.GEOMETRY):
        for metric in ("corr", "mi"):
            reg = sitk.ImageRegistrationMethod()
            if metric == "corr":
                reg.SetMetricAsCorrelation()
            else:
                reg.SetMetricAsMattesMutualInformation(32)
                reg.SetMetricSamplingStrategy(reg.RANDOM)
                reg.SetMetricSamplingPercentage(0.2, seed=1)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 400)
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetShrinkFactorsPerLevel([4, 2, 1])
            reg.SetSmoothingSigmasPerLevel([2, 1, 0])
            reg.SetInitialTransform(sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), init),
                                    inPlace=False)
            try:
                t = reg.Execute(fixed, moving)
            except RuntimeError:
                continue
            m = N.to_array(sitk.Resample(moving, fixed, t, sitk.sitkLinear, 0.0))
            r = np.corrcoef(f[head], m[head])[0, 1]
            if np.isfinite(r) and r > best_r:
                best, best_r = t, r
    return best, best_r


def resample(moving, fixed, t):
    import SimpleITK as sitk
    return N.to_array(sitk.Resample(moving, fixed, t, sitk.sitkBSpline, 0.0))


def slab_blur(high, low, inplane=1.0, slab=1.0):
    """`high` (SimpleITK) blurred to `low`'s voxel: Gaussian in-plane (FWHM = `inplane` voxels) and a box `slab`
    slices long along whichever of high's axes lies closest to low's slice axis."""
    import SimpleITK as sitk
    x = sitk.GetArrayFromImage(high).astype(np.float32)                     # (z, y, x) order
    d_low = np.array(low.GetDirection()).reshape(3, 3)[:, 2]               # low's slice direction
    d_high = np.array(high.GetDirection()).reshape(3, 3)
    ax_high = int(np.argmax(np.abs(d_high.T @ d_low)))                      # 0, 1, 2 = high's x, y, z
    sp_high = np.array(high.GetSpacing())
    sp_low = np.array(low.GetSpacing())
    sigma = np.zeros(3)
    box = np.ones(3, int)
    for a in range(3):
        if a == ax_high:
            box[a] = max(1, int(round(slab * sp_low[2] / sp_high[a])))
        else:
            sigma[a] = inplane * sp_low[0] / 2.355 / sp_high[a]
    x = ndi.gaussian_filter(x, sigma[::-1])
    x = ndi.uniform_filter(x, box[::-1])
    out = sitk.GetImageFromArray(x)
    out.CopyInformation(high)
    return out


def class_means(v, k=4, iters=50):
    """Sorted means of k intensity classes (k-means): noise moves voxels between classes but barely moves the means."""
    c = np.percentile(v, np.linspace(10, 90, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(v[:, None] - c[None]), 1)
        c = np.array([v[lab == i].mean() if np.any(lab == i) else c[i] for i in range(k)])
    return np.sort(c)


def paired_class_means(source, target, mask, k=4):
    """(source means, target means) over the same voxels: classes found on the source (k-means; 3 T has little
    noise), each eroded to its pure core, and both images averaged there. Pairing by voxel, not by rank, keeps a
    class that exists in one image only (vessels at 3 T) from shifting the others; averaging pure voxels keeps
    partial volume and noise from pulling the means together."""
    c = class_means(source[mask], k)
    lab = np.full(source.shape, -1)
    lab[mask] = np.argmin(np.abs(source[mask][:, None] - c[None]), 1)
    src, dst = [], []
    for i in range(k):
        pure = ndi.binary_erosion(lab == i)
        if pure.sum() >= 50:
            src.append(source[pure].mean())
            dst.append(target[pure].mean())
    order = np.argsort(src)
    return np.array(src)[order], np.array(dst)[order]


def piecewise(x, src, dst):
    """Piecewise-linear map through (0, 0) and the class means (src -> dst), extended linearly past the last one."""
    s, d = np.concatenate([[0], src]), np.concatenate([[0], dst])
    out = np.interp(x, s, d)
    hi = x > s[-1]
    return np.where(hi, d[-1] + (x - s[-1]) * (d[-1] - d[-2]) / max(s[-1] - s[-2], 1e-6), out)


def add_noise(clean, wm_signal, snr=SNR, smoothing_px=SMOOTHING_PX, seed=0):
    """clean + Gaussian noise of SD wm_signal / snr, blurred in-plane (axes 0 and 1) by smoothing_px pixels."""
    n = np.random.default_rng(seed).normal(0, 1, clean.shape)
    if smoothing_px:
        n = ndi.gaussian_filter(n, (smoothing_px, smoothing_px, 0))
        n /= n.std()
    return clean + (wm_signal / snr) * n


def spin_echo(tissue, tr=TR_S, te=TE_S):
    t1, t2 = T1T2_50MT[tissue]
    return PD[tissue] * (1 - np.exp(-tr / t1)) * np.exp(-te / t2)


def tissue_fractions(x, voxel_mm):
    """{tissue: fraction map} of a T2w volume x: white matter, grey matter and CSF means from three k-means classes
    of the brain, air or bone at 0, and each voxel split linearly between the two neighbouring means (above the CSF
    mean, all CSF; the scalp's fat lands there too)."""
    brain = N.white_matter(x, voxel_mm)[0]
    v = x[brain]
    c = np.percentile(v, [20, 50, 95])
    for _ in range(50):
        lab = np.argmin(np.abs(v[:, None] - c[None]), 1)
        c = np.array([v[lab == k].mean() for k in range(3)])
    anchors = np.concatenate([[0.0], np.sort(c)])
    names = ("air", "wm", "gm", "csf")
    f = {n: np.zeros_like(x) for n in names}
    xc = np.clip(x, 0, anchors[-1])
    for i in range(3):
        lo, hi = anchors[i], anchors[i + 1]
        m = (xc >= lo) & (xc <= hi) if i == 0 else (xc > lo) & (xc <= hi)
        w = (xc[m] - lo) / (hi - lo)
        f[names[i]][m] += 1 - w
        f[names[i + 1]][m] += w
    return f


def blurred_fractions(high, low_grid, t, inplane=1.0, slab=1.0):
    """Tissue fractions of the 3 T image, blurred to the 64 mT voxel and resampled with transform t."""
    import SimpleITK as sitk
    f = tissue_fractions(N.to_array(high), high.GetSpacing())
    out = {}
    for name in ("wm", "gm", "csf"):
        img = sitk.GetImageFromArray(f[name].transpose(2, 1, 0).astype(np.float32))
        img.CopyInformation(high)
        out[name] = resample(slab_blur(img, low_grid, inplane, slab), low_grid, t)
    return out


def stages(high, low_grid, contrast_from, wm, snr=SNR, smoothing_px=SMOOTHING_PX, seed=0):
    """({stage: fake 64 mT array}, Dice of the fake's and the grid's head masks) for 3 T image `high` on SimpleITK
    image `low_grid`'s grid, with contrast learned from `contrast_from` (another 64 mT scan of the same person) and
    white-matter mask `wm` on the low grid. Every stage is scaled to that scan's white matter."""
    blurred = slab_blur(high, low_grid)
    t, _ = register_transform(low_grid, blurred)
    raw = resample(high, low_grid, t)
    res = resample(blurred, low_grid, t)
    t_ref, _ = register_transform(low_grid, contrast_from)
    ref = resample(contrast_from, low_grid, t_ref)
    head_res, head_ref = N.head_mask(res), N.head_mask(ref)
    deep = N.white_matter(ref, low_grid.GetSpacing())[0]            # brain: the head more than 15 mm deep
    con = np.zeros_like(res)
    for region in (deep, head_res & ~deep):                         # brain and the rest (scalp, fat, bone) apart
        both = region & head_res & head_ref
        src, dst = paired_class_means(res, ref, both)
        con[region] = piecewise(res[region], src, dst)
    fr = blurred_fractions(high, low_grid, t)
    lit = sum(spin_echo(k) * fr[k] for k in fr)
    fit_mask = deep & head_ref
    A = np.stack([fr[k][fit_mask] for k in fr], 1)
    w = np.linalg.lstsq(A, ref[fit_mask], rcond=None)[0]
    fitted = sum(wk * fr[k] for wk, k in zip(w, fr))
    scale = lambda x: x * ref[wm].mean() / x[wm].mean() * head_res
    out = {STAGES[0]: scale(raw), STAGES[1]: scale(res), STAGES[2]: scale(con), STAGES[3]: scale(lit),
           STAGES[4]: scale(fitted)}
    out[STAGES[5]] = add_noise(out[STAGES[4]], ref[wm].mean(), snr, smoothing_px, seed) * head_res
    grid_head = N.head_mask(N.to_array(low_grid))
    dice = 2 * np.sum(head_res & grid_head) / (head_res.sum() + grid_head.sum())
    return out, dice, dict(zip(fr, w / w[0]))
