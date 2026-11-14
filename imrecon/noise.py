"""Noise in reconstructed MR images: how much there is, and how it is shaped.

Vendor images rarely allow the textbook background estimate. GE's 3 T images quantize the air around the head to 0
or 1, and Hyperfine's 64 mT images mask most of it and are reconstructed with a learned denoiser. So noise is
measured inside the brain, in white matter, three ways:

    laplacian_sigma   Immerkaer's 3 x 3 Laplacian stencil on each in-plane slice, over the flattest voxels. Exact for
                      white noise; reads low for spatially correlated noise, since the stencil only sees high
                      frequencies. Anatomy left in the flattest voxels reads high.
    rescan_sigma      two scans of the same person: register, match intensities, and take the robust SD of the
                      difference / sqrt(2). Sees noise at every frequency, plus whatever registration and different
                      slice placement leave behind.
    noise_spectrum    the in-plane power spectrum of that difference: flat for white noise, falling for smoothed noise.

A noise model a simulator can draw from has three parameters per scan: the SD, the in-plane smoothing (the Gaussian
width that turns white noise into noise with the measured spectrum, `fit_smoothing`), and the SNR they give against
white matter. The Laplacian and rescan SDs bracket the true one: the first misses smoothed noise, the second adds
whatever registration and different slice placement leave, minus what interpolating one scan smooths away.
"""

import numpy as np
from scipy import ndimage as ndi


def otsu(v, bins=256):
    """Otsu threshold of the values v."""
    h, e = np.histogram(v, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def head_mask(x, axis=2):
    """The head: above Otsu's threshold, closed, and filled slice by slice along `axis` (a 3D fill leaks out where
    the field of view cuts the head)."""
    m = ndi.binary_closing(x > otsu(x[x > 0]), iterations=3)
    m = np.moveaxis(m, axis, -1)
    m = np.stack([ndi.binary_fill_holes(m[..., s]) for s in range(m.shape[-1])], -1)
    return np.moveaxis(m, -1, axis)


def white_matter(x, voxel_mm, depth_mm=15.0):
    """(brain, wm, class means): brain is the head more than depth_mm from its surface (past scalp and skull); its
    voxels split into three intensity classes by k-means, and on T2w the darkest is white matter. The WM mask is
    eroded by one voxel so it stays off tissue edges."""
    brain = ndi.distance_transform_edt(head_mask(x), sampling=voxel_mm) > depth_mm
    v = x[brain]
    c = np.percentile(v, [20, 50, 90])
    for _ in range(50):
        lab = np.argmin(np.abs(v[:, None] - c[None]), 1)
        c = np.array([v[lab == k].mean() for k in range(3)])
    wm = np.zeros_like(brain)
    wm[brain] = lab == 0
    return brain, ndi.binary_erosion(wm), c


LAPLACIAN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)


def laplacian_residuals(x, mask, axis=2, keep=0.5):
    """(residuals, slice index) over the in-plane slices (planes perpendicular to `axis`): Immerkaer's 3 x 3
    Laplacian stencil divided by 6, so white noise of SD s gives residuals of SD s, at the `keep` share of mask
    voxels with the smallest local gradient."""
    x, mask = np.moveaxis(x, axis, -1), np.moveaxis(mask, axis, -1)
    r, g, sl = [], [], []
    for s in range(x.shape[2]):
        m = mask[..., s]
        if not m.any():
            continue
        im = x[..., s]
        sm = ndi.gaussian_filter(im, 1.0)
        r.append(ndi.convolve(im, LAPLACIAN, mode="nearest")[m] / 6)
        g.append(np.hypot(ndi.sobel(sm, 0), ndi.sobel(sm, 1))[m])
        sl.append(np.full(int(m.sum()), s))
    r, g, sl = np.concatenate(r), np.concatenate(g), np.concatenate(sl)
    sel = g <= np.quantile(g, keep)
    return r[sel], sl[sel]


def laplacian_sigma(x, mask, axis=2, keep=0.5):
    """Immerkaer's noise SD: sqrt(pi / 2) times the mean absolute Laplacian residual (laplacian_residuals)."""
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(laplacian_residuals(x, mask, axis, keep)[0])))


def robust_sd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def register(fixed, moving):
    """`moving` resampled onto `fixed` (SimpleITK images) by a rigid registration on correlation, B-spline."""
    import SimpleITK as sitk
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS), inPlace=False)
    t = reg.Execute(fixed, moving)
    return sitk.Resample(moving, fixed, t, sitk.sitkBSpline, 0.0), -reg.GetMetricValue()


def to_array(img):
    """SimpleITK image to a numpy array in (x, y, z) order, as nibabel gives it."""
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype(float)


def rescan_sigma(a, b, mask):
    """(per-scan noise SD, difference) of two registered scans a and b (arrays), b scaled to a's mean in the mask."""
    d = a - b * a[mask].mean() / b[mask].mean()
    return robust_sd(d[mask]) / np.sqrt(2), d


def noise_spectrum(d, mask, axis=2, k=16):
    """(radial frequency bin edges, relative power) of the in-plane power spectrum of d over k x k patches lying
    wholly inside mask, Hann-windowed; power is relative to the 0.1-0.2 cycles/pixel band (white noise: 1 at every
    frequency)."""
    d, mask = np.moveaxis(d, axis, -1), np.moveaxis(mask, axis, -1)
    w = np.outer(np.hanning(k), np.hanning(k))
    p = []
    for s in range(d.shape[2]):
        for i in range(0, d.shape[0] - k + 1, k // 2):
            for j in range(0, d.shape[1] - k + 1, k // 2):
                if mask[i:i + k, j:j + k, s].all():
                    q = d[i:i + k, j:j + k, s]
                    p.append(np.abs(np.fft.fft2((q - q.mean()) * w)) ** 2)
    if not p:
        return None, None
    ps = np.mean(p, 0)
    f = np.fft.fftfreq(k)
    fr = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.71])
    band = [ps[(fr > lo) & (fr <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
    return edges, np.array(band) / band[1]


def high_frequency_share(rel):
    """Power above 0.3 cycles/pixel relative to 0.1-0.2 (noise_spectrum's last three bands); white noise gives 1."""
    return float(np.mean(rel[3:]))


def fit_smoothing(rel, edges=None):
    """In-plane Gaussian width (pixels) that best turns white noise into the relative spectrum `rel` from
    noise_spectrum: a Gaussian filter of width s passes power exp(-(2 pi f s)^2). Fitted on log power over the
    0.2-0.5 cycles/pixel bands, where anatomy left in a rescan difference matters least; 0 means white."""
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.71]) if edges is None else edges
    f = (edges[:-1] + edges[1:]) / 2
    ref, use = f[1], slice(2, 5)
    widths = np.linspace(0, 3, 301)
    model = lambda s: -(2 * np.pi * s) ** 2 * (f[use] ** 2 - ref ** 2)
    err = [np.sum((np.log(np.maximum(rel[use], 1e-6)) - model(s)) ** 2) for s in widths]
    return float(widths[int(np.argmin(err))])
