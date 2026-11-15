"""Which distribution the noise follows: sweep each candidate's SD, measure surprisal, keep the minimum.

Surprisal is the mean of -log2 p(x) over the samples, in bits per sample: the lower, the better the distribution
predicts the data. Every residual family has mean zero and is parametrized by its SD, so one sweep over SD serves
them all; the SD at a family's minimum is its maximum-likelihood SD. Shape parameters (Student-t degrees of freedom,
generalized-normal exponent, skewness) are separate family members. The sweep runs on one half of the samples and
the surprisal reported is on the other half, so extra shape freedom cannot win by fitting noise in the sample.

Residual samples should be centred and divided by a robust SD (noise.robust_sd) first, so SDs are in units of it and
surprisals compare across scans.

Three kinds of sample, three sets of families:

    residuals   zero-centred noise (Laplacian residuals, rescan differences): FAMILIES, symmetric and skewed
    magnitude   raw intensities in small flat patches, one unknown true signal per patch: MAGNITUDE_FAMILIES, the
                distributions of |signal + complex noise| (Gaussian approximation, Rician for one coil, noncentral
                chi for L coils combined by root-sum-of-squares); they differ only where SNR is low
    background  air, where the true signal is 0: BACKGROUND_FAMILIES (Rayleigh, chi, half-normal, gamma, ...)
"""

import numpy as np
from scipy import special, stats


def _gennorm(beta):
    def logpdf(x, sd):
        a = sd * np.sqrt(special.gamma(1 / beta) / special.gamma(3 / beta))
        return stats.gennorm.logpdf(x, beta, scale=a)
    return logpdf


def _t(nu):
    def logpdf(x, sd):
        return stats.t.logpdf(x, nu, scale=sd * np.sqrt((nu - 2) / nu))
    return logpdf


def _standardized(dist):
    """logpdf(x, sd) of a scipy distribution shifted and scaled to mean 0 and this SD."""
    m, v = dist.stats("mv")
    def logpdf(x, sd):
        k = sd / np.sqrt(v)
        return dist.logpdf(x / k + m) - np.log(k)
    return logpdf


_skewnorm = lambda a: _standardized(stats.skewnorm(a))
_nig = lambda a: _standardized(stats.norminvgauss(a, 0))
_johnsonsu = lambda d: _standardized(stats.johnsonsu(0, d))

FAMILIES = {
    "Gaussian": lambda x, sd: stats.norm.logpdf(x, scale=sd),
    "Laplace": lambda x, sd: stats.laplace.logpdf(x, scale=sd / np.sqrt(2)),
    "logistic": lambda x, sd: stats.logistic.logpdf(x, scale=sd * np.sqrt(3) / np.pi),
    **{f"Student-t nu={n}": _t(n) for n in (3, 4, 6, 10, 30)},
    **{f"gen. normal beta={b}": _gennorm(b) for b in (0.75, 1.5, 3.0, 4.0)},
    "hyperbolic secant": lambda x, sd: stats.hypsecant.logpdf(x, scale=sd * 2 / np.pi),
    "Cauchy (SD = scale)": lambda x, sd: stats.cauchy.logpdf(x, scale=sd),
    **{f"skew-normal a={a}": _skewnorm(a) for a in (-2, -1, 1, 2)},
    **{f"NIG a={a}": _nig(a) for a in (1.0, 3.0)},
    **{f"Johnson SU d={d}": _johnsonsu(d) for d in (1.0, 2.0)},
}

SD_GRID = np.geomspace(0.3, 3.0, 161)


def surprisal(x, family, sd):
    """Mean surprisal (bits per sample) of samples x under `family` with this SD."""
    return float(-np.mean(FAMILIES[family](x, sd)) / np.log(2))


def sweep(x, family, sds=SD_GRID):
    """Surprisal at every SD in sds."""
    return np.array([surprisal(x, family, s) for s in sds])


def fit(train, test, families=None, sds=SD_GRID):
    """{family: (SD at the minimum surprisal on train, surprisal on test at that SD, minimum on train)}."""
    out = {}
    for f in families or FAMILIES:
        curve = sweep(train, f, sds)
        i = int(np.argmin(curve))
        out[f] = (float(sds[i]), surprisal(test, f, sds[i]), float(curve[i]))
    return out


def split(samples, slices, max_n=50000, seed=0):
    """(train, test): even and odd slices, each subsampled to at most max_n. Neighbouring voxels share noise when
    it is smoothed, so the halves are split by slice, not voxel by voxel."""
    rng = np.random.default_rng(seed)
    out = []
    for parity in (0, 1):
        v = samples[slices % 2 == parity]
        out.append(v if len(v) <= max_n else rng.choice(v, max_n, replace=False))
    return out


# magnitude: x = |nu + noise|, noise complex Gaussian of SD sigma per component, from L coils combined as root sum of
# squares (L = 1: Rician). Every family sets each patch's true signal nu so that its own mean equals the patch mean,
# so the families differ only in shape.
_T = np.concatenate([np.linspace(0, 20, 2001), np.geomspace(20.01, 1e4, 400)])     # nu / sigma


def _mean_over_sigma(L):
    """E[x] / sigma as a function of nu / sigma (_T), for L coils."""
    t = _T
    m = np.sqrt(2) * np.exp(special.gammaln(L + 0.5) - special.gammaln(L)) * special.hyp1f1(-0.5, L, -t ** 2 / 2)
    return np.where(t > 30, np.sqrt(t ** 2 + (2 * L - 1)), m)     # asymptote, where hyp1f1 loses precision


_MEAN = {L: _mean_over_sigma(L) for L in (1, 2, 4, 8)}
def _nc_chi(L):
    def logpdf(x, nu, sigma):
        if L == 1:
            return stats.rice.logpdf(x, nu / sigma, scale=sigma)
        u = (x / sigma) ** 2
        return stats.ncx2.logpdf(u, 2 * L, (nu / sigma) ** 2) + np.log(2 * x / sigma ** 2)
    return logpdf, L


MAGNITUDE_FAMILIES = {
    "Gaussian (patch mean)": (lambda x, nu, sigma: stats.norm.logpdf(x, nu, sigma), 0),
    "Rician": _nc_chi(1),
    **{f"noncentral chi, {L} coils": _nc_chi(L) for L in (2, 4, 8)},
}


def magnitude_surprisal(patches, family, sigma):
    """Mean surprisal (bits per voxel) of patches (n, k) under a magnitude family with noise SD sigma."""
    logpdf, L = MAGNITUDE_FAMILIES[family]
    m = patches.mean(1, keepdims=True)
    nu = m if L == 0 else sigma * np.interp(m / sigma, _MEAN[L], _T)    # below the noise floor's mean: nu = 0
    return float(-np.mean(logpdf(np.maximum(patches, 1e-9), nu, sigma)) / np.log(2))


def magnitude_fit(train, test, sigmas):
    """{family: (sigma at the minimum surprisal on train, surprisal on test there)}."""
    out = {}
    for f in MAGNITUDE_FAMILIES:
        curve = np.array([magnitude_surprisal(train, f, s) for s in sigmas])
        i = int(np.argmin(curve))
        out[f] = (float(sigmas[i]), magnitude_surprisal(test, f, sigmas[i]))
    return out


# background: true signal 0. Each entry: scipy distribution for a scale (and shape)
BACKGROUND_FAMILIES = {
    "Rayleigh (Rician, 1 coil)": lambda sc: stats.rayleigh(scale=sc),
    **{f"chi, {L} coils": (lambda L: lambda sc: stats.chi(2 * L, scale=sc))(L) for L in (2, 4, 8)},
    "Maxwell": lambda sc: stats.maxwell(scale=sc),
    "half-normal": lambda sc: stats.halfnorm(scale=sc),
    "exponential": lambda sc: stats.expon(scale=sc),
    **{f"gamma k={k}": (lambda k: lambda sc: stats.gamma(k, scale=sc))(k) for k in (2, 4, 8)},
    **{f"lognormal s={s}": (lambda s: lambda sc: stats.lognorm(s, scale=sc))(s) for s in (0.25, 0.5, 1.0)},
}


def background_fit(train, test):
    """{family: (SD of the fitted distribution, surprisal on test)}: each family's scale swept over a grid around the
    data's RMS, minimum surprisal on train."""
    scales = np.sqrt(np.mean(train ** 2)) * np.geomspace(0.03, 3, 241)
    out = {}
    for f, make in BACKGROUND_FAMILIES.items():
        curve = np.array([-np.mean(make(sc).logpdf(train)) / np.log(2) for sc in scales])
        i = int(np.argmin(curve))
        d = make(scales[i])
        out[f] = (float(d.std()), float(-np.mean(d.logpdf(test)) / np.log(2)))
    return out
