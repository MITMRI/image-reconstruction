"""One shareable figure: the measured 64 mT noise against the fitted distributions (numbers only, no images of people).

    python scripts/noise_fit_figure.py path/to/ds006557 [--out results/figures]

Noise samples are ds006557's 64 mT scan-rescan differences / sqrt(2) in white matter (two Hyperfine scanners, 23
people), each person's in units of its robust SD, pooled. Every family's SD is swept on even slices and the curve at
the minimum surprisal is drawn; the legend gives the held-out surprisal (odd slices) above the best. Writes
12_noise_fit.png (best fit, Gaussian, Laplace) and 13_noise_fit_all.png (every family, one panel each).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import datasets as DS  # noqa: E402
from imrecon import distributions as Dist  # noqa: E402
from imrecon import noise as N  # noqa: E402

SHOW = ("Student-t nu=10", "Gaussian", "Laplace")


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    from noise_report import AQUA, BLUE, INK2, ORANGE, style
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)
    xs, sls = [], []
    for a, b in DS.rescan_pairs(DS.ds006557(args.root)):
        ia, ib = (sitk.ReadImage(str(s["path"]), sitk.sitkFloat32) for s in (a, b))
        xa = N.to_array(ia)
        _, wm, _ = N.white_matter(xa, ia.GetSpacing())
        _, d = N.rescan_sigma(xa, N.to_array(N.register(ia, ib)[0]), wm)
        v = d[wm] / np.sqrt(2)
        xs.append((v - np.median(v)) / N.robust_sd(v))
        sls.append(np.nonzero(wm)[2])
        print(a["subject"], flush=True)
    v, sl = np.concatenate(xs), np.concatenate(sls)
    train, test = Dist.split(v, sl, max_n=400000)
    fits = Dist.fit(train, test)
    best = min(fits, key=lambda f: fits[f][1])
    style(plt)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    edges = np.linspace(-6, 6, 121)
    h, e = np.histogram(test, edges, density=True)
    mid = (e[:-1] + e[1:]) / 2
    xx = np.linspace(-6, 6, 600)
    for a, logy in zip(ax, (False, True)):
        a.bar(mid, h, width=np.diff(e), color="#d9d8d3", edgecolor="#d9d8d3", label="measured 64 mT noise")
        for f, col in zip(SHOW, (ORANGE, BLUE, AQUA)):
            a.plot(xx, np.exp(Dist.FAMILIES[f](xx, fits[f][0])), color=col, lw=1.8,
                   label=f"{f.replace('nu=', 'ν = ')}: +{fits[f][1] - fits[best][1]:.3f} bits")
        a.set_xlabel("noise / robust SD")
        a.set_ylabel("density (log scale)" if logy else "density")
        if logy:
            a.set_yscale("log")
            a.set_ylim(h[h > 0].min() * 0.5, h.max() * 2)
            a.set_title("Tails", loc="left")
        else:
            a.set_xlim(-4.5, 4.5)
            a.set_title("64 mT noise is nearly Gaussian", loc="left")
            a.legend(fontsize=8, loc="upper left")
    fig.text(0.01, 0.005, f"Scan-rescan differences / √2 in white matter: 23 people, two Hyperfine scanners, "
             f"{len(v):,} voxels. Each curve at the SD of its lowest surprisal; bits = held-out surprisal above "
             f"the best fit.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "12_noise_fit.png", dpi=160)
    order = sorted(fits, key=lambda f: fits[f][1])
    cols = 5
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), sharex=True, sharey=True)
    for a, f in zip(axes.ravel(), order):
        a.bar(mid, h, width=np.diff(e), color="#d9d8d3", edgecolor="#d9d8d3")
        a.plot(xx, np.exp(Dist.FAMILIES[f](xx, fits[f][0])), color=ORANGE if f == best else BLUE, lw=1.6)
        a.set_yscale("log")
        a.set_ylim(h[h > 0].min() * 0.5, h.max() * 3)
        a.set_title(f"{f.replace('nu=', 'ν = ').replace('beta=', 'β = ')}\n+{fits[f][1] - fits[best][1]:.3f} bits, "
                    f"SD {fits[f][0]:.2f}", loc="left", fontsize=8)
        a.tick_params(labelsize=7)
    for a in axes.ravel()[len(order):]:
        a.set_visible(False)
    fig.suptitle("64 mT noise (grey) against every distribution tried, best first (log density; noise / robust SD)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.003, f"Scan-rescan differences / √2 in white matter: 23 people, two Hyperfine scanners, "
             f"{len(v):,} voxels. Each curve at the SD of its lowest surprisal; bits = held-out surprisal above "
             f"the best fit.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / "13_noise_fit_all.png", dpi=150)
    print(f"wrote {out / '12_noise_fit.png'}, 13_noise_fit_all.png; best {best}")


if __name__ == "__main__":
    main()
