"""Tables and figures from scripts/field_noise.py's output (numbers only, never images of people).

    python scripts/noise_report.py [results]

Prints the summary tables and writes PNG figures to <results>/figures:

    1_snr.png               white-matter SNR per scan and scanner; 64 mT rescan SNR against the Laplacian's
    2_spectrum.png          in-plane spectrum of the 64 mT rescan difference, against white noise
    3_surprisal_sweep.png   surprisal against SD for each family, per kind of sample: the sweep and its minima
    4_family_ranking.png    held-out surprisal of every family against the Gaussian, across scans
    5_background.png        the air around the head, where the vendor left it: background families
    6_controls.png          noise / white matter against brain volume, by scanner; scanner effects
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import distributions as Dist  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"                  # categorical slots 1-3 (dataviz palette)
FIELD_COLOR = {3.0: BLUE, 0.064: ORANGE}


def read(path):
    import csv
    def num(v):
        try:
            return float(v)
        except ValueError:
            return {"True": True, "False": False}.get(v, v)
    with open(path) as f:
        return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 2,
                         "legend.frameon": False})


def plain_log(ax, axis, ticks):
    """Log axis with plain-number ticks (no 2 x 10^1 clutter)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def label(r):
    return f"{r['scanner']} ({r['dataset']})"


def fig_snr(scans, pairs, out, plt):
    groups = sorted({label(r) for r in scans}, key=lambda g: (-[r for r in scans if label(r) == g][0]["field_T"], g))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, g in enumerate(groups):
        rs = [r for r in scans if label(r) == g]
        snr = np.array([r["wm"] / r["sigma_lap"] for r in rs])
        jitter = (np.random.default_rng(i).random(len(snr)) - 0.5) * 0.35
        ax[0].scatter(snr, i + jitter, s=22, color=FIELD_COLOR[rs[0]["field_T"]], edgecolor=SURFACE, linewidth=1)
        ax[0].plot([np.median(snr)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax[0].text(np.percentile(snr, 90) * 1.12, i + 0.32, f"median {np.median(snr):.0f}, n = {len(snr)}",
                   va="center", fontsize=8, color=INK2)
    ax[0].set_yticks(range(len(groups)), groups)
    ax[0].set_xscale("log")
    plain_log(ax[0], "x", [5, 10, 20, 50, 100, 200, 500])
    ax[0].set_xlabel("white-matter SNR per voxel (Laplacian estimate)")
    ax[0].set_title("SNR by scanner (blue 3 T, orange 64 mT)", loc="left")
    ax[0].grid(axis="y", visible=False)
    for same, marker, name in ((False, "o", "two scanners (ds006557)"), (True, "s", "one scanner, two sessions (LUMC)")):
        ps = [p for p in pairs if bool(p["same_scanner"]) == same]
        lap = []
        for p in ps:
            a = [r for r in scans if r["dataset"] == p["dataset"] and r["subject"] == p["subject"] and r["field_T"] < 1]
            lap.append(np.mean([r["wm"] / r["sigma_lap"] for r in a]))
        ax[1].scatter(lap, [p["snr_rescan"] for p in ps], s=30, marker=marker, color=ORANGE if same else AQUA,
                      edgecolor=SURFACE, linewidth=1, label=name)
    lim = [4, 40]
    ax[1].plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax[1].text(22, 25, "equal", color=MUTED, fontsize=8, rotation=45)
    ax[1].set(xlim=lim, ylim=lim, xscale="log", yscale="log", xlabel="64 mT SNR from one scan (Laplacian)",
              ylabel="64 mT SNR from the rescan difference")
    plain_log(ax[1], "x", [5, 10, 20, 40])
    plain_log(ax[1], "y", [5, 10, 20, 40])
    ax[1].set_title("One-scan SNR overstates 64 mT", loc="left")
    ax[1].legend(loc="upper left", fontsize=8, markerscale=1)
    fig.tight_layout()
    fig.savefig(out / "1_snr.png", dpi=150)


def fig_spectrum(pairs, out, plt):
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.71])
    f = (edges[:-1] + edges[1:]) / 2
    rel = np.array([[float(v) for v in p["spectrum"].split()] for p in pairs if isinstance(p.get("spectrum"), str)])
    sm = np.median([p["smoothing_px"] for p in pairs if isinstance(p.get("smoothing_px"), float)])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in rel:
        ax.plot(f, r, color=GRID, lw=1)
    ax.plot(f, np.median(rel, 0), color=ORANGE, marker="o", ms=5, label=f"64 mT rescan difference, median of {len(rel)}")
    ff = np.linspace(0.05, 0.6, 100)
    ax.plot(ff, np.exp(-(2 * np.pi * sm) ** 2 * (ff ** 2 - f[1] ** 2)), color=INK2, lw=1.5, ls="--",
            label=f"white noise blurred by a {sm:.2f}-pixel Gaussian")
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.text(0.58, 1.12, "white noise", color=MUTED, fontsize=8, ha="right")
    ax.set(yscale="log", xlabel="in-plane spatial frequency (cycles / pixel)",
           ylabel="noise power, relative to 0.1-0.2 cycles / pixel")
    ax.set_title("64 mT noise is smoothed", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "2_spectrum.png", dpi=150)


KINDS = [("residual", 3.0, "3 T, Laplacian residuals"), ("residual", 0.064, "64 mT, Laplacian residuals"),
         ("rescan", 0.064, "64 mT, rescan difference"), ("magnitude", 3.0, "3 T, flat-patch magnitudes"),
         ("magnitude", 0.064, "64 mT, flat-patch magnitudes")]


def ids_for(kind, field, scans, pairs):
    if kind == "rescan":
        return [p["id"] for p in pairs]
    return [r["id"] for r in scans if r["field_T"] == field]


def fig_sweep(curves, scans, pairs, out, plt):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), sharey=False)
    for ax, (kind, field, title) in zip(axes, KINDS):
        fams = list(Dist.MAGNITUDE_FAMILIES if kind == "magnitude" else Dist.FAMILIES)
        ids = [i for i in ids_for(kind, field, scans, pairs) if f"{kind}|{i}|{fams[0]}" in curves]
        if not ids:
            ax.set_visible(False)
            continue
        x = Dist.SD_GRID if kind != "magnitude" else np.geomspace(0.3, 3, 121)
        floor = np.array([min(curves[f"{kind}|{i}|{g}"].min() for g in fams) for i in ids])[:, None]
        rel = {fam: np.median(np.array([curves[f"{kind}|{i}|{fam}"] for i in ids]) - floor, 0) for fam in fams}
        best = min(rel, key=lambda k: rel[k].min())
        show = [best] + [g for g in (fams[0], "Laplace" if kind != "magnitude" else "Rician") if g != best][:2]
        for fam in fams:
            if fam not in show:
                ax.plot(x, rel[fam], color=GRID, lw=1)
        for fam, col in zip(show, (ORANGE, BLUE, AQUA)):
            ax.plot(x, rel[fam], color=col, label=fam)
            i = int(np.argmin(rel[fam]))
            ax.scatter([x[i]], [rel[fam][i]], color=col, s=36, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xscale("log")
        plain_log(ax, "x", [0.5, 1, 2])
        ax.set(ylim=(-0.02, 0.6), xlabel="SD swept (units of robust SD)" if kind != "magnitude"
               else "noise SD swept (units of patch SD)")
        ax.set_title(f"{title} (n = {len(ids)})", loc="left", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper center")
    axes[0].set_ylabel("surprisal above the best fit (bits / sample)")
    fig.suptitle("Surprisal sweep: each family's SD swept, minimum marked (median over scans; grey = other families)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "3_surprisal_sweep.png", dpi=150)


def fig_ranking(scans, pairs, out, plt):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.4))
    for ax, (kind, field, title) in zip(axes, KINDS):
        rows = pairs if kind == "rescan" else [r for r in scans if r["field_T"] == field]
        fams = list(Dist.MAGNITUDE_FAMILIES if kind == "magnitude" else Dist.FAMILIES)
        ref = fams[0]
        rows = [r for r in rows if isinstance(r.get(f"{kind}|{ref}|bits"), float)]
        if not rows:
            ax.set_visible(False)
            continue
        d = {f: np.array([r[f"{kind}|{f}|bits"] - r[f"{kind}|{ref}|bits"] for r in rows]) for f in fams}
        order = sorted(fams, key=lambda f: np.median(d[f]))
        for y, f in enumerate(order):
            q = np.percentile(d[f], [25, 50, 75])
            col = ORANGE if y == 0 else (BLUE if f == ref else INK2)
            ax.plot(q[[0, 2]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.scatter([q[1]], [y], color=col, s=26, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=MUTED, lw=1, ls=":")
        ax.set_yticks(range(len(order)), order, fontsize=7.5)
        ax.invert_yaxis()
        lo, hi = np.percentile(np.concatenate(list(d.values())), [2, 98])
        ax.set_xlim(min(lo, -0.01) - 0.01, min(hi, 0.5) + 0.01)
        ax.set_xlabel(f"held-out bits / sample vs {ref.split(' (')[0]} (left = better)")
        ax.set_title(f"{title} (n = {len(rows)})", loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Which distribution fits: median and quartiles over scans", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "4_family_ranking.png", dpi=150)


def fig_background(scans, out, plt):
    fams = list(Dist.BACKGROUND_FAMILIES)
    rows = [r for r in scans if isinstance(r.get(f"background|{fams[0]}|bits"), float)]
    if not rows:
        return None
    best = np.array([min(r[f"background|{f}|bits"] for f in fams) for r in rows])
    d = {f: np.array([r[f"background|{f}|bits"] for r in rows]) - best for f in fams}
    order = sorted(fams, key=lambda f: np.median(d[f]))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for y, f in enumerate(order):
        q = np.percentile(d[f], [25, 50, 75])
        col = ORANGE if y == 0 else (BLUE if f.startswith("Rayleigh") else INK2)
        ax.plot(q[[0, 2]], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.scatter([q[1]], [y], color=col, s=26, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("held-out bits / sample above the best family for that scan")
    groups = sorted({label(r) for r in rows})
    ax.set_title(f"Air around the head ({len(rows)} scans: {', '.join(groups)})", loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "5_background.png", dpi=150)
    return rows


def fig_controls(scans, fit, out, plt):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for field in (3.0, 0.064):
        rs = [r for r in scans if r["field_T"] == field]
        ax[0].scatter([r["brain_ml"] for r in rs], [100 * r["sigma_lap"] / r["wm"] for r in rs], s=20,
                      color=FIELD_COLOR[field], edgecolor=SURFACE, linewidth=1, label=f"{field:g} T")
    ax[0].set(yscale="log", xlabel="brain volume (ml; head > 15 mm deep)", ylabel="noise / white matter (%)")
    ax[0].set_title("Body size barely moves the noise", loc="left")
    ax[0].legend(fontsize=8)
    names = [n for n in fit if n != "intercept"]
    for y, n in enumerate(names):
        b, lo, hi = (100 * (np.exp(v) - 1) for v in fit[n])
        col = ORANGE if n.startswith("scanner") else BLUE
        ax[1].plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax[1].scatter([b], [y], color=col, s=28, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax[1].axvline(0, color=MUTED, lw=1, ls=":")
    ax[1].set_yticks(range(len(names)), names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("% change in 64 mT noise / white matter (95 % bootstrap interval)")
    ax[1].set_title("64 mT: scanner against person (one model)", loc="left")
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "6_controls.png", dpi=150)


def ols(y, X, names, n_boot=2000, seed=0):
    X = np.column_stack([np.ones(len(y))] + X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    boot = [np.linalg.lstsq(X[i], y[i], rcond=None)[0] for i in (rng.integers(len(y), size=len(y)) for _ in range(n_boot))]
    lo, hi = np.percentile(boot, [2.5, 97.5], 0)
    return {n: (b, l, h) for n, b, l, h in zip(["intercept"] + names, beta, lo, hi)}


def tables(scans, pairs):
    q = lambda v: f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f}]"
    print("white-matter SNR per scan (Laplacian), median [quartiles]")
    for g in sorted({label(r) for r in scans}):
        rs = [r for r in scans if label(r) == g]
        print(f"  {g:48s} n {len(rs):3d}  voxel {rs[0]['voxel_mm3']:5.2f} mm3  SNR {q([r['wm'] / r['sigma_lap'] for r in rs])}")
    for same, name in ((False, "ds006557, two scanners"), (True, "LUMC, one scanner, two sessions")):
        ps = [p for p in pairs if bool(p["same_scanner"]) == same]
        sm = [p["smoothing_px"] for p in ps if isinstance(p.get("smoothing_px"), float)]
        print(f"  64 mT rescan, {name:34s} n {len(ps):3d}  SNR {q([p['snr_rescan'] for p in ps])}  smoothing "
              f"{np.median(sm):.2f} px  registration r {np.median([p['registration_corr'] for p in ps]):.3f}")
    lumc = [r for r in scans if r["dataset"] == "lumc"]
    per = []
    for s in sorted({r["subject"] for r in lumc if r["field_T"] == 3.0}):
        hi = [r for r in lumc if r["subject"] == s and r["field_T"] == 3.0]
        lo = [r for r in lumc if r["subject"] == s and r["field_T"] < 1]
        if hi and lo:
            a, b = hi[0], sorted(lo, key=lambda r: r["session"])[0]
            per.append(((a["wm"] / a["sigma_lap"]) / np.sqrt(a["voxel_mm3"]), (b["wm"] / b["sigma_lap"]) / np.sqrt(b["voxel_mm3"])))
    if per:
        per = np.array(per)
        print(f"  LUMC same people, SNR per sqrt(mm3) (Laplacian): 3 T Philips {q(per[:, 0])}, 64 mT {q(per[:, 1])}, "
              f"ratio {q(per[:, 0] / per[:, 1])} (n = {len(per)}; the Laplacian flatters smoothed 64 mT noise)")


def controls(scans):
    lo = [r for r in scans if r["field_T"] < 1]
    scanners = sorted({r["scanner"] for r in lo})
    ref = max(scanners, key=lambda s: sum(r["scanner"] == s for r in lo))
    y = np.log([r["sigma_lap"] / r["wm"] for r in lo])
    bv = np.array([r["brain_ml"] for r in lo])
    X = [(bv - np.median(bv)) / 100] + [np.array([r["scanner"] == s for r in lo], float) for s in scanners if s != ref]
    names = ["brain volume (per 100 ml)"] + [f"scanner {s.split()[-1]} vs {ref.split()[-1]}" for s in scanners if s != ref]
    fit = ols(y, X, names)
    print(f"\n64 mT noise / white matter (log; Laplacian) against scanner and brain volume, n = {len(lo)} scans; "
          f"% change, 95 % bootstrap interval")
    for n, (b, l, h) in fit.items():
        if n != "intercept":
            print(f"  {n:40s} {100 * (np.exp(b) - 1):+6.1f} % [{100 * (np.exp(l) - 1):+.1f}, {100 * (np.exp(h) - 1):+.1f}]")
    ds = [r for r in lo if r["dataset"] == "ds006557"]
    y2 = np.log([r["sigma_lap"] / r["wm"] for r in ds])
    X2 = [np.array([r["age"] for r in ds]) / 10 - 4.5, np.array([r["sex"] == "M" for r in ds], float),
          (np.array([r["brain_ml"] for r in ds]) - np.median(bv)) / 100, np.array([r["session"] == "HFE" for r in ds], float)]
    fit2 = ols(y2, X2, ["age (per decade)", "male", "brain volume (per 100 ml)", "scanner HFE vs HFC"])
    print(f"  ds006557 only, with age and sex (n = {len(ds)} scans of {len(ds) // 2} people):")
    for n, (b, l, h) in fit2.items():
        if n != "intercept":
            print(f"    {n:38s} {100 * (np.exp(b) - 1):+6.1f} % [{100 * (np.exp(l) - 1):+.1f}, {100 * (np.exp(h) - 1):+.1f}]")
    return fit


def shapes(scans, pairs):
    print("\nbest distribution by held-out surprisal (bits / sample against the first family; scans where best)")
    for kind, field, title in KINDS:
        rows = pairs if kind == "rescan" else [r for r in scans if r["field_T"] == field]
        fams = list(Dist.MAGNITUDE_FAMILIES if kind == "magnitude" else Dist.FAMILIES)
        rows = [r for r in rows if isinstance(r.get(f"{kind}|{fams[0]}|bits"), float)]
        if not rows:
            continue
        b = np.array([[r[f"{kind}|{f}|bits"] for f in fams] for r in rows])
        wins = np.bincount(b.argmin(1), minlength=len(fams))
        order = np.argsort(np.median(b - b[:, :1], 0))[:4]
        print(f"  {title} (n = {len(rows)}): " + ";  ".join(
            f"{fams[i]} {np.median(b[:, i] - b[:, 0]):+.4f} (best in {wins[i]})" for i in order))
    fams = list(Dist.BACKGROUND_FAMILIES)
    rows = [r for r in scans if isinstance(r.get(f"background|{fams[0]}|bits"), float)]
    if rows:
        b = np.array([[r[f"background|{f}|bits"] for f in fams] for r in rows])
        wins = np.bincount(b.argmin(1), minlength=len(fams))
        order = np.argsort(np.median(b - b.min(1, keepdims=True), 0))[:5]
        print(f"  background (n = {len(rows)}): " + ";  ".join(
            f"{fams[i]} +{np.median(b[:, i] - b.min(1)):.3f} (best in {wins[i]})" for i in order)
              + f";  Rayleigh +{np.median(b[:, 0] - b.min(1)):.3f}")


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = Path((argv or sys.argv[1:] or ["results"])[0])
    scans, pairs = read(res / "scans.csv"), read(res / "pairs.csv")
    z = np.load(res / "curves.npz", allow_pickle=True)
    curves = dict(zip(z["keys"], z["curves"]))
    out = res / "figures"
    out.mkdir(exist_ok=True)
    style(plt)
    tables(scans, pairs)
    fit = controls(scans)
    shapes(scans, pairs)
    fig_snr(scans, pairs, out, plt)
    fig_spectrum(pairs, out, plt)
    fig_sweep(curves, scans, pairs, out, plt)
    fig_ranking(scans, pairs, out, plt)
    fig_background(scans, out, plt)
    fig_controls(scans, fit, out, plt)
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
