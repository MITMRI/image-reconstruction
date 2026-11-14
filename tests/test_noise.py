"""Checks of the estimators on plain random arrays, where the answer is known."""

import numpy as np
from scipy import ndimage as ndi

from imrecon import distributions as Dist
from imrecon import noise as N


def _flat(shape=(96, 96, 12), seed=0):
    return np.full(shape, 100.0), np.random.default_rng(seed)


def test_laplacian_sigma_white_and_smoothed():
    x, rng = _flat()
    mask = np.zeros(x.shape, bool)
    mask[8:-8, 8:-8] = True
    n = rng.normal(0, 1, x.shape)
    assert abs(N.laplacian_sigma(x + 5 * n, mask) - 5) < 0.15         # exact for white noise
    s = ndi.gaussian_filter(n, (1, 1, 0))
    s *= 5 / s.std()
    assert N.laplacian_sigma(x + s, mask) < 1.5                        # blind to smoothed noise of the same SD


def test_rescan_sigma_and_spectrum():
    x, rng = _flat()
    mask = np.zeros(x.shape, bool)
    mask[8:-8, 8:-8] = True
    a, b = x + 4 * rng.normal(0, 1, x.shape), x + 4 * rng.normal(0, 1, x.shape)
    s, d = N.rescan_sigma(a, b, mask)
    assert abs(s - 4) < 0.1
    _, rel = N.noise_spectrum(d, mask)
    assert abs(N.high_frequency_share(rel) - 1) < 0.15 and N.fit_smoothing(rel) < 0.15
    n = ndi.gaussian_filter(rng.normal(0, 1, x.shape), (0.7, 0.7, 0))
    _, rel = N.noise_spectrum(n, mask)
    assert abs(N.fit_smoothing(rel) - 0.7) < 0.15                     # recovers the smoothing width


def test_surprisal_sweep_finds_family_and_sd():
    rng = np.random.default_rng(1)
    g = rng.normal(0, 1, 40000)
    fits = Dist.fit(g[:20000], g[20000:])
    assert abs(fits["Gaussian"][0] - 1) < 0.03
    best = min(fits, key=lambda f: fits[f][1])
    assert fits[best][1] - fits["Gaussian"][1] > -0.002                # nothing beats the true family by more than noise
    t = rng.standard_t(4, 40000) / np.sqrt(2)                          # unit-SD Student-t, 4 degrees of freedom
    fits = Dist.fit(t[:20000], t[20000:])
    assert min(fits, key=lambda f: fits[f][1]).startswith("Student-t")
    assert fits["Gaussian"][1] - fits["Student-t nu=4"][1] > 0.05
