"""Two pictures behind the noise numbers, written to results/figures (never committed: one shows a person's scans).

    python scripts/noise_examples.py --ds006557 path/to/ds006557 --lumc path/to/lumc/v5/Data [--subject sub-HYPE01]

7_scans.png          one person: 3 T, 64 mT, the 64 mT rescan (registered), 64 mT minus 3 T (3 T resampled onto the
                     64 mT grid and scaled to its white matter), and 64 mT minus its rescan, which is mostly noise
8_distributions.png  the measured noise as it is (histogram, pooled over scans, each scan in units of its robust SD)
                     with every family drawn at the SD where its surprisal is lowest; log scale shows the tails
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import datasets as DS  # noqa: E402
from imrecon import distributions as Dist  # noqa: E402
from imrecon import noise as N  # noqa: E402
from noise_report import AQUA, BLUE, ORANGE, style  # noqa: E402


def load(path):
    import SimpleITK as sitk
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def fig_scans(root, sid, out, plt):
    import SimpleITK as sitk
    s = {r["session"]: r for r in DS.ds006557(root) if r["subject"] == sid}
    a, b, hi = load(s["HFC"]["path"]), load(s["HFE"]["path"]), load(s["GE"]["path"])
    xa = N.to_array(a)
    _, wm, c = N.white_matter(xa, a.GetSpacing())
    xb = N.to_array(N.register(a, b)[0])
    xb *= xa[wm].mean() / xb[wm].mean()
    hi_on_a, _ = N.register(a, sitk.SmoothingRecursiveGaussian(hi, [v / 2.355 for v in a.GetSpacing()]))
    xh = N.to_array(hi_on_a)                           # 3 T blurred to a 64 mT voxel, on the 64 mT grid
    xh *= xa[wm].mean() / np.median(xh[wm])
    z = xa.shape[2] // 2
    vmax = np.percentile(xa[wm], 99) * 3
    d_field, d_rescan = xa - xh, xa - xb
    lim = 4 * N.robust_sd(d_rescan[wm])
    panels = [(xh, "3 T, blurred to the 64 mT voxel", "gray", 0, vmax),
              (xa, "64 mT, scanner C", "gray", 0, vmax), (xb, "64 mT, scanner E (registered)", "gray", 0, vmax),
              (d_field, "64 mT - 3 T: contrast + resolution + noise", "RdBu_r", -lim, lim),
              (d_rescan, "64 mT - rescan: mostly noise", "RdBu_r", -lim, lim)]
    fig, ax = plt.subplots(1, 5, figsize=(17, 4.4))
    for axi, (im, title, cmap, lo, hi_) in zip(ax, panels):
        h = axi.imshow(np.rot90(im[:, :, z]), cmap=cmap, vmin=lo, vmax=hi_)
        axi.set_title(title, fontsize=8.5, loc="left")
        axi.axis("off")
        if cmap != "gray":
            fig.colorbar(h, ax=axi, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(f"{sid}, axial slice {z}: difference colour scale +-4 noise SDs (rescan estimate)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "7_scans.png", dpi=150)


def pooled(kind, scans, pairs):
    """Samples of one kind from many scans, each centred and in units of its robust SD, with slice indices."""
    xs, sls = [], []
    if kind[0] == "rescan":
        for a, b in pairs:
            ia, ib = load(a["path"]), load(b["path"])
            xa = N.to_array(ia)
            _, wm, _ = N.white_matter(xa, ia.GetSpacing())
            _, d = N.rescan_sigma(xa, N.to_array(N.register(ia, ib)[0]), wm)
            xs.append(d[wm] / np.sqrt(2))
            sls.append(np.nonzero(wm)[2])
    else:
        for s in scans:
            img = load(s["path"])
            x = N.to_array(img)
            if kind[0] == "background":
                v, sl = N.background(x, img.GetSpacing())
                if len(v) < 2000 or np.median(v) < 5:
                    continue
                xs.append(v / np.sqrt(np.mean(v ** 2)))        # background is not centred: scale only
                sls.append(sl)
                continue
            _, wm, _ = N.white_matter(x, img.GetSpacing())
            v, sl = N.laplacian_residuals(x, wm)
            xs.append(v)
            sls.append(sl)
    if kind[0] != "background":
        xs = [(v - np.median(v)) / N.robust_sd(v) for v in xs]
    rng = np.random.default_rng(0)
    take = [rng.choice(len(v), min(len(v), 20000), replace=False) for v in xs]
    return np.concatenate([v[t] for v, t in zip(xs, take)]), np.concatenate([s[t] for s, t in zip(sls, take)])


def fig_distributions(scans, pairs, out, plt):
    kinds = [(("residual", 3.0), "3 T, Laplacian residuals"), (("residual", 0.064), "64 mT, Laplacian residuals"),
             (("rescan", 0.064), "64 mT, rescan difference / sqrt(2)"), (("background", 0.064), "64 mT, air")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.6), gridspec_kw={"height_ratios": [2, 1]})
    for col, (kind, title) in enumerate(kinds):
        sel = [s for s in scans if s["field_T"] == kind[1]]
        v, sl = pooled(kind, sel, pairs)
        train, test = Dist.split(v, sl, max_n=200000)
        if kind[0] == "background":
            fits = Dist.background_fit(train, test)
            scales = np.sqrt(np.mean(train ** 2)) * np.geomspace(0.03, 3, 241)
            dists = {}
            for f, make in Dist.BACKGROUND_FAMILIES.items():
                curve = [-np.mean(make(sc).logpdf(train)) for sc in scales]
                dists[f] = make(scales[int(np.argmin(curve))])
            pdf = {f: d.pdf for f, d in dists.items()}
            bits = {f: fits[f][1] for f in fits}
            edges = np.linspace(0, np.percentile(v, 99.9), 120)
            ref = "Rayleigh (Rician, 1 coil)"
        else:
            fits = Dist.fit(train, test)
            pdf = {f: (lambda f, sd: lambda x: np.exp(Dist.FAMILIES[f](x, sd)))(f, fits[f][0]) for f in fits}
            bits = {f: fits[f][1] for f in fits}
            edges = np.linspace(-6, 6, 161)
            ref = "Gaussian"
        order = sorted(bits, key=bits.get)
        show = [order[0]] + [f for f in (ref, order[1]) if f != order[0]][:2]
        h, e = np.histogram(test, edges, density=True)
        mid = (e[:-1] + e[1:]) / 2
        xx = np.linspace(edges[0], edges[-1], 600)[1:]
        for row, logy in ((0, True), (1, False)):
            ax = axes[row, col]
            ax.bar(mid, h, width=np.diff(e), color="#d9d8d3", edgecolor="#d9d8d3", label="measured (held-out half)")
            for f in order[::-1]:
                if f not in show:
                    ax.plot(xx, pdf[f](xx), color="#c9c8c2", lw=0.8)
            for f, colr in zip(show, (ORANGE, BLUE, AQUA)):
                ax.plot(xx, pdf[f](xx), color=colr, lw=1.8,
                        label=f"{f}: {bits[f] - bits[order[0]]:+.3f} bits" if row == 0 else None)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(max(h[h > 0].min() * 0.3, 1e-6), h.max() * 3)
                ax.set_title(f"{title} (n = {len(v):,} samples)", loc="left", fontsize=9)
                ax.legend(fontsize=7.5, loc="upper right" if kind[0] != "background" else "upper right")
            ax.set_xlabel("value / robust SD" if kind[0] != "background" else "value / RMS")
        axes[0, col].set_ylabel("density (log)")
        axes[1, col].set_ylabel("density")
    fig.suptitle("Noise as measured, with each family at the SD of its lowest surprisal (grey = other families; "
                 "bits = held-out surprisal above the best)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "8_distributions.png", dpi=150)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds006557", required=True)
    ap.add_argument("--lumc")
    ap.add_argument("--subject", default="sub-HYPE01")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style(plt)
    fig_scans(Path(args.ds006557), args.subject, out, plt)
    scans = DS.ds006557(args.ds006557) + (DS.lumc(args.lumc) if args.lumc else [])
    fig_distributions(scans, DS.rescan_pairs(scans), out, plt)
    print(f"wrote {out}/7_scans.png, 8_distributions.png")


if __name__ == "__main__":
    main()
