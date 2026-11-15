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
    2c fitted      the same fractions, with each tissue's 64 mT signal taken from another 64 mT scan of the same person
                   (tissue_signals: class means). Scalp is a fifth tissue, because the 3 T scan is fat-suppressed
    2d T1w labels  as 2c, but grey matter, white matter and dark tissue from the 3 T T1w scan (MPRAGE, registered
                   onto the T2w), where white matter is bright, grey matter between and CSF dark, and iron does not
                   darken deep grey matter as it does on 3 T T2w. The T2w then splits the dark part: bright on T2w is
                   CSF, dark on both is bone or air
    3 noise        on 2d (2c without a T1w): Student-t (nu = 10), blurred in-plane by `smoothing_px`, SD = white matter / `snr` (the measured medians:
                   SNR 8.3 from rescans, 0.6 px); the air outside the head is set to 0, as Hyperfine's masking leaves it

The noise is the best-fitting family from the rescan differences, Student-t with nu = 10 (a Gaussian is within
0.002-0.006 bits per voxel of it).
"""

import numpy as np
from scipy import ndimage as ndi

from . import noise as N

SNR = 8.3
SMOOTHING_PX = 0.6
NU = 10                                              # Student-t: best family in most 64 mT rescan pairs
STAGES = ("0 resampled", "1 + resolution", "2a + contrast: map", "2b + contrast: physics", "2c + contrast: fitted",
          "2d + contrast: T1w labels", "3 + noise")
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


def add_noise(clean, wm_signal, snr=SNR, smoothing_px=SMOOTHING_PX, nu=NU, seed=0):
    """clean + noise of SD wm_signal / snr: Student-t with nu degrees of freedom (the best fit to 64 mT rescan
    differences; None for Gaussian), blurred in-plane (axes 0 and 1) by smoothing_px pixels."""
    rng = np.random.default_rng(seed)
    n = rng.standard_t(nu, clean.shape) if nu else rng.normal(0, 1, clean.shape)
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
    mean, all CSF). Scalp is its own tissue: the head outside what Otsu's threshold calls head, found with a much
    lower threshold, since the 3 T CUBE scan suppresses fat and leaves the scalp dark."""
    brain = N.white_matter(x, voxel_mm)[0]
    v = x[brain]
    c = np.percentile(v, [20, 50, 95])
    for _ in range(50):
        lab = np.argmin(np.abs(v[:, None] - c[None]), 1)
        c = np.array([v[lab == k].mean() for k in range(3)])
    anchors = np.concatenate([[0.0], np.sort(c)])
    names = ("air", "wm", "gm", "csf")
    f = {n: np.zeros_like(x) for n in names + ("scalp",)}
    xc = np.clip(x, 0, anchors[-1])
    for i in range(3):
        lo, hi = anchors[i], anchors[i + 1]
        m = (xc >= lo) & (xc <= hi) if i == 0 else (xc > lo) & (xc <= hi)
        w = (xc[m] - lo) / (hi - lo)
        f[names[i]][m] += 1 - w
        f[names[i + 1]][m] += w
    loose = ndi.binary_closing(x > 0.1 * anchors[1], iterations=3)
    loose = np.stack([ndi.binary_fill_holes(loose[..., k]) for k in range(loose.shape[2])], -1)
    scalp = loose & ~N.head_mask(x)
    for n in names:
        f[n][scalp] = 0
    f["scalp"][scalp] = 1
    return f


def kmeans_1d(v, k, init):
    c = np.percentile(v, init)
    for _ in range(50):
        lab = np.argmin(np.abs(v[:, None] - c[None]), 1)
        c = np.array([v[lab == i].mean() if np.any(lab == i) else c[i] for i in range(k)])
    return np.sort(c)


def tissue_fractions_t1(t2, t1, voxel_mm):
    """{tissue: fraction map} from a T1w volume t1 on the grid of the T2w volume t2: CSF, grey and white matter means
    from three k-means classes of the T1w brain, each voxel split linearly between neighbouring means (below the CSF
    mean all 'dark', above white matter all white matter); the dark part is CSF in proportion to T2w brightness
    (up to the T2w CSF mean) and bone or air for the rest. Scalp as in tissue_fractions."""
    brain = N.white_matter(t2, voxel_mm)[0]
    dark, gm, wm = kmeans_1d(t1[brain], 3, [10, 45, 80])
    csf_t2 = kmeans_1d(t2[brain], 3, [20, 50, 95])[-1]
    x = np.clip(t1, dark, wm)
    f_gm = np.where(x <= gm, (x - dark) / (gm - dark), (wm - x) / (wm - gm))
    f_wm = np.where(x > gm, (x - gm) / (wm - gm), 0.0)
    f_dark = np.where(x <= gm, (gm - x) / (gm - dark), 0.0)
    share = np.clip(t2 / csf_t2, 0, 1)
    head = N.head_mask(t2)
    f = {"wm": f_wm * head, "gm": f_gm * head, "csf": f_dark * share * head}
    base = tissue_fractions(t2, voxel_mm)
    f["scalp"] = base["scalp"]
    for k in ("wm", "gm", "csf"):
        f[k][f["scalp"] > 0] = 0
    return f


def blurred_fractions(high, low_grid, t, inplane=1.0, slab=1.0, t1=None):
    """Tissue fractions of the 3 T image (from T2w alone, or with a T1w SimpleITK image on any grid), blurred to the
    64 mT voxel and resampled with transform t."""
    import SimpleITK as sitk
    if t1 is None:
        f = tissue_fractions(N.to_array(high), high.GetSpacing())
    else:
        # same session, same scanner coordinates: resample, do not register (a correlation-picked registration
        # between inverted contrasts chose a 3.5 mm shift)
        f = tissue_fractions_t1(N.to_array(high), resample(t1, high, sitk.Euler3DTransform()), high.GetSpacing())
    out = {}
    for name in ("wm", "gm", "csf", "scalp"):
        img = sitk.GetImageFromArray(f[name].transpose(2, 1, 0).astype(np.float32))
        img.CopyInformation(high)
        out[name] = resample(slab_blur(img, low_grid, inplane, slab), low_grid, t)
    return out


def tissue_signals(ref, brain, scalp):
    """64 mT signal of each tissue, from another 64 mT scan `ref` of the same person: white matter, grey matter and
    CSF as the means of four k-means classes of its brain (white matter the darkest holding a tenth of it, grey
    matter the next, CSF the brightest; the third is grey matter and CSF mixed), the scalp as its mean where the
    3 T scan puts scalp. Class means are not pulled toward each other the way a least-squares fit of overlapping
    blurred fraction maps is (that fit gave grey matter = white matter)."""
    v = ref[brain]
    c = np.percentile(v, [5, 25, 55, 90])
    for _ in range(50):
        lab = np.argmin(np.abs(v[:, None] - c[None]), 1)
        c = np.array([v[lab == k].mean() if np.any(lab == k) else c[k] for k in range(4)])
    share = np.bincount(lab, minlength=4) / len(v)
    order = [i for i in np.argsort(c) if share[i] >= 0.1 or c[i] == c.max()]
    return {"wm": c[order[0]], "gm": c[order[1]], "csf": c.max(),
            "scalp": ref[scalp].mean() if scalp.any() else c[order[0]]}


def stages(high, low_grid, contrast_from, wm, snr=SNR, smoothing_px=SMOOTHING_PX, seed=0, t1=None):
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
    lit = sum(spin_echo(k) * fr[k] for k in ("wm", "gm", "csf"))
    sig = tissue_signals(ref, deep & head_ref, fr["scalp"] > 0.7)
    fitted = sum(sig[k] * fr[k] for k in fr)
    lit = lit + fr["scalp"] * sig["scalp"] * spin_echo("wm") / sig["wm"]         # no scalp in the 50 mT table
    scale = lambda x: x * ref[wm].mean() / x[wm].mean() * head_res
    out = {STAGES[0]: scale(raw), STAGES[1]: scale(res), STAGES[2]: scale(con), STAGES[3]: scale(lit),
           STAGES[4]: scale(fitted)}
    last = STAGES[4]
    if t1 is not None:
        fr1 = blurred_fractions(high, low_grid, t, t1=t1)
        sig1 = tissue_signals(ref, deep & head_ref, fr1["scalp"] > 0.7)
        out[STAGES[5]] = scale(sum(sig1[k] * fr1[k] for k in fr1))
        last = STAGES[5]
    out[STAGES[6]] = add_noise(out[last], ref[wm].mean(), snr, smoothing_px, seed=seed) * head_res
    grid_head = N.head_mask(N.to_array(low_grid))
    dice = 2 * np.sum(head_res & grid_head) / (head_res.sum() + grid_head.sum())
    return out, dice, {k: sig[k] / sig["wm"] for k in sig}
