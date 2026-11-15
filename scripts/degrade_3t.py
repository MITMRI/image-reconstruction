"""Fake 64 mT scans from 3 T, built up one step at a time (imrecon.degrade), against the real ones.

    python scripts/degrade_3t.py path/to/ds006557 [--out results] [--show sub-HYPE01]

Per ds006557 person, the 3 T scan is moved onto scanner C's grid and degraded in steps (resampled; + resolution;
+ contrast learned from scanner E's scan of the same person; + noise), and each step is compared with scanner C's real
scan, which the degrader never sees. Scanner E's scan, registered onto C, is the yardstick: real minus rescan holds two
noises plus registration error, so a perfect fake would differ from the real scan by the same amount.

    ratio       robust SD of real - fake over robust SD of real - rescan, in white matter and in the brain: 1 = matched
    beyond      the part of real - fake not explained by noise, sqrt(SD(real - fake)^2 - SD(real - rescan)^2)
    bias        mean real - fake in each quarter of the brain's intensities (darkest = white matter, brightest = CSF),
                as % of white matter: the contrast error. Quarters are set on the rescan, whose noise is independent
                of the real scan's and the fake's (set on the real scan, its own noise would bias every quarter)
    SNR, spectrum, shape   of the full fake against the real scan, by the estimators in imrecon.noise

Writes results/degrade.csv, figures/9_fake_vs_real.png (one person: never committed) and 10_degrade_summary.png.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import datasets as DS  # noqa: E402
from imrecon import degrade as G  # noqa: E402
from imrecon import noise as N  # noqa: E402

QUARTERS = ("darkest (WM)", "2nd", "3rd", "brightest (CSF)")


def load(path):
    import SimpleITK as sitk
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def person(by, keep=None):
    c, e, g = (load(by[k]["path"]) for k in ("HFC", "HFE", "GE"))
    real = N.to_array(c)
    brain, wm, cls = N.white_matter(real, c.GetSpacing())
    t, _ = G.register_transform(c, e)
    rescan = G.resample(e, c, t)
    rescan *= real[wm].mean() / rescan[wm].mean()
    fakes, dice = G.stages(g, c, e, wm, seed=0)
    row = {"head_dice": dice, "fake_corr": np.corrcoef(real[brain], fakes[G.STAGES[2]][brain])[0, 1],
           "rescan_corr": np.corrcoef(real[brain], rescan[brain])[0, 1], "wm_signal": cls[0]}
    d_rescan = real - rescan
    q = np.quantile(rescan[brain], [0.25, 0.5, 0.75])    # quarters by the rescan: its noise is independent of
    quarter = np.digitize(rescan, q)                       # both the real scan's and the fake's
    for i, name in enumerate(QUARTERS):
        row[f"bias_rescan_{name}"] = float(np.mean(d_rescan[brain & (quarter == i)]) / cls[0])
    for region, m in (("wm", wm), ("brain", brain)):
        row[f"sd_rescan_{region}"] = N.robust_sd(d_rescan[m])
    for st, fake in fakes.items():
        d = real - fake
        for region, m in (("wm", wm), ("brain", brain)):
            sd = N.robust_sd(d[m])
            row[f"ratio_{st}_{region}"] = sd / row[f"sd_rescan_{region}"]
            row[f"beyond_{st}_{region}"] = np.sqrt(max(sd ** 2 - row[f"sd_rescan_{region}"] ** 2, 0)) / cls[0]
        for i, name in enumerate(QUARTERS):
            row[f"bias_{st}_{name}"] = float(np.mean(d[brain & (quarter == i)]) / cls[0])
    full = fakes[G.STAGES[-1]]
    row["snr_lap_real"] = cls[0] / N.laplacian_sigma(real, wm)
    row["snr_lap_fake"] = cls[0] / N.laplacian_sigma(full, wm)
    for k, d in (("rescan", d_rescan), ("fake", real - full)):
        _, rel = N.noise_spectrum(d, brain)
        row[f"smoothing_{k}"] = N.fit_smoothing(rel) if rel is not None else np.nan
    if keep is not None:
        keep.update(real=real, rescan=rescan, fakes=fakes, wm=wm, brain=brain)
    return row


def fig_person(sid, k, out, plt):
    z = k["real"].shape[2] // 2
    vmax = np.percentile(k["real"][k["wm"]], 99) * 3
    lim = 4 * N.robust_sd((k["real"] - k["rescan"])[k["wm"]])
    fig, ax = plt.subplots(2, 5, figsize=(17, 8))
    top = [(k["real"], "real 64 mT (scanner C)"), (k["rescan"], "rescan (scanner E, registered)")] + \
          [(k["fakes"][s], f"fake: {s}") for s in G.STAGES[1:]]
    bottom = [(k["real"] - k["rescan"], "real - rescan: the target")] + \
             [(k["real"] - k["fakes"][s], f"real - fake {s}") for s in G.STAGES]
    for a, (im, title) in zip(ax[0], top):
        a.imshow(np.rot90(im[:, :, z]), cmap="gray", vmin=0, vmax=vmax)
        a.set_title(title, fontsize=8.5, loc="left")
        a.axis("off")
    for a, (im, title) in zip(ax[1], bottom):
        h = a.imshow(np.rot90(im[:, :, z]), cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set_title(title, fontsize=8.5, loc="left")
        a.axis("off")
    fig.colorbar(h, ax=ax[1].tolist(), fraction=0.012, pad=0.01).ax.tick_params(labelsize=7)
    fig.suptitle(f"{sid}, slice {z}: building the fake one step at a time (differences on a +-4 noise-SD scale)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.savefig(out / "9_fake_vs_real.png", dpi=150, bbox_inches="tight")


def fig_summary(rows, out, plt):
    from noise_report import BLUE, INK2, MUTED, ORANGE, SURFACE
    g = lambda k: np.array([r[k] for r in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for j, (region, col, name) in enumerate((("wm", ORANGE, "white matter"), ("brain", BLUE, "whole brain"))):
        for i, st in enumerate(G.STAGES):
            v = g(f"ratio_{st}_{region}")
            x = i + (j - 0.5) * 0.3 + (np.random.default_rng(i).random(len(v)) - 0.5) * 0.12
            ax[0].scatter(x, v, s=16, color=col, edgecolor=SURFACE, linewidth=0.8, label=name if i == 0 else None)
            ax[0].plot([i + (j - 0.5) * 0.3 - 0.1, i + (j - 0.5) * 0.3 + 0.1], [np.median(v)] * 2, color=INK2, lw=2)
    ax[0].axhline(1, color=MUTED, lw=1, ls="--")
    ax[0].text(3.45, 1.03, "matched", color=MUTED, fontsize=8, ha="right")
    ax[0].set_xticks(range(len(G.STAGES)), G.STAGES)
    ax[0].set_ylabel("SD(real - fake) / SD(real - rescan)")
    ax[0].set_yscale("log")
    ax[0].set_title("How far each step gets (1 = as close as a rescan)", loc="left")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="x", visible=False)
    for i, st in enumerate(G.STAGES[1:3]):
        med = [np.median(g(f"bias_{st}_{n}")) * 100 for n in QUARTERS]
        ax[1].plot(range(4), med, marker="o", ms=5, color=(MUTED, ORANGE)[i], label=f"fake {st}")
    med = [np.median(g(f"bias_rescan_{n}")) * 100 for n in QUARTERS]
    ax[1].plot(range(4), med, marker="o", ms=5, color=BLUE, label="rescan (the target)")
    ax[1].axhline(0, color=MUTED, lw=1, ls=":")
    ax[1].set_xticks(range(4), QUARTERS)
    ax[1].set_xlabel("quarter of the brain's intensities (set on the rescan)")
    ax[1].set_ylabel("mean real - fake (% of white matter)")
    ax[1].set_title("Contrast error, median over people", loc="left")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "10_degrade_summary.png", dpi=150)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from noise_report import style
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--out", default="results")
    ap.add_argument("--show", default="sub-HYPE01")
    args = ap.parse_args(argv)
    scans = DS.ds006557(args.root)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    style(plt)
    rows = []
    for sid in sorted({s["subject"] for s in scans}):
        by = {s["session"]: s for s in scans if s["subject"] == sid}
        keep = {} if sid == args.show else None
        r = {"subject": sid, **person(by, keep)}
        rows.append(r)
        if keep:
            fig_person(sid, keep, out / "figures", plt)
        print(f"{sid}: Dice {r['head_dice']:.3f}, brain r fake {r['fake_corr']:.2f} rescan {r['rescan_corr']:.2f} | WM ratio " + " ".join(
            f"{r[f'ratio_{s}_wm']:.2f}" for s in G.STAGES) + " | brain " + " ".join(
            f"{r[f'ratio_{s}_brain']:.2f}" for s in G.STAGES), flush=True)
    with open(out / "degrade.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    g = lambda k: np.array([r[k] for r in rows], float)
    med = lambda v: f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}]"
    print(f"\n{len(rows)} people, median [quartiles]; SD(real - fake) / SD(real - rescan):")
    for st in G.STAGES:
        print(f"  {st:16s} white matter {med(g(f'ratio_{st}_wm'))}   brain {med(g(f'ratio_{st}_brain'))}   "
              f"bias by quarter (% WM) " + ", ".join(f"{np.median(g(f'bias_{st}_{n}')) * 100:+.0f}" for n in QUARTERS))
    print(f"  {'rescan':16s} bias by quarter (% WM) " + ", ".join(
        f"{np.median(g(f'bias_rescan_{n}')) * 100:+.0f}" for n in QUARTERS))
    print(f"  brain correlation with the real scan: rescan {med(g('rescan_corr'))}, fake {med(g('fake_corr'))}")
    print(f"  Laplacian SNR: real {med(g('snr_lap_real'))}, fake {med(g('snr_lap_fake'))}")
    print(f"  spectrum smoothing (px): real - rescan {med(g('smoothing_rescan'))}, real - fake {med(g('smoothing_fake'))}")
    fig_summary(rows, out / "figures", plt)
    print(f"wrote {out}/degrade.csv, figures/9_fake_vs_real.png, figures/10_degrade_summary.png")


if __name__ == "__main__":
    main()
