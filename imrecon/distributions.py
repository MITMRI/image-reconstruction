"""Which distribution the noise follows: sweep each candidate's SD, measure surprisal, keep the minimum.

Surprisal is the mean of -log2 p(x) over the samples, in bits per sample: the lower, the better the distribution
predicts the data. Every family here is symmetric about zero and parametrized by its SD, so one sweep over SD
serves them all; the SD at a family's minimum is its maximum-likelihood SD. Shape parameters (Student-t degrees of
freedom, generalized-normal exponent) are separate family members. The sweep runs on one half of the samples and
the surprisal reported is on the other half, so extra shape freedom cannot win by fitting noise in the sample.

Samples should be centred and divided by a robust SD (noise.robust_sd) first, so SDs are in units of it and
surprisals compare across scans.
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


FAMILIES = {
    "Gaussian": lambda x, sd: stats.norm.logpdf(x, scale=sd),
    "Laplace": lambda x, sd: stats.laplace.logpdf(x, scale=sd / np.sqrt(2)),
    "logistic": lambda x, sd: stats.logistic.logpdf(x, scale=sd * np.sqrt(3) / np.pi),
    **{f"Student-t nu={n}": _t(n) for n in (3, 4, 6, 10, 30)},
    **{f"gen. normal beta={b}": _gennorm(b) for b in (0.75, 1.5, 3.0, 4.0)},
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
