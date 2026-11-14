"""Noise at 64 mT against 3 T in the same people: OpenNeuro ds006557 (Vasa et al. 2025), T2w.

    python scripts/paired_field_noise.py path/to/ds006557 [--out results/ds006557_noise.csv] [--subjects 5]

Each subject has a 3 T GE CUBE T2w (1 mm isotropic) and two 64 mT Hyperfine axial T2w scans (1.5 x 1.5 x 5 mm) on
two identical scanners, days apart. Per subject, in white matter (imrecon.noise):

    3 T     WM signal and the Laplacian noise SD (the only estimate a single scan allows)
    64 mT   WM signal, the Laplacian SD of each scan, the scan-rescan SD, the in-plane spectrum of the rescan
            difference (over the brain) and the Gaussian smoothing width that fits it
    shape   which distribution the noise samples follow (imrecon.distributions: sweep each family's SD, take the
            minimum surprisal, compare on held-out slices): 3 T and 64 mT Laplacian residuals, 64 mT rescan
            difference / sqrt(2)
    person  age and sex (participants.tsv) and brain volume (the 3 T head more than 15 mm deep), to control the noise figures for
            body size and to separate the two 64 mT scanners from the people in them

Writes one CSV row per subject (numbers only; no images leave the data folder) and prints the medians: the
parameters of a noise model for each field.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import distributions as Dist  # noqa: E402
from imrecon import noise as N  # noqa: E402

LOW_VOXEL_MM3 = 1.5 * 1.5 * 5.0


def shape_fits(row, name, samples, slices):
    """Distribution fits of one sample set, centred and in units of its robust SD, into row."""
    v = (samples - np.median(samples)) / N.robust_sd(samples)
    for fam, (sd, bits, _) in Dist.fit(*Dist.split(v, slices)).items():
        row[f"bits_{name}_{fam}"] = bits
        row[f"sd_{name}_{fam}"] = sd


def subject(root, sid, people):
    import SimpleITK as sitk
    p = {"3T": root / sid / "ses-GE/anat" / f"{sid}_ses-GE_T2w.nii.gz",
         "HFC": root / sid / "ses-HFC/anat" / f"{sid}_ses-HFC_acq-axi_T2w.nii.gz",
         "HFE": root / sid / "ses-HFE/anat" / f"{sid}_ses-HFE_acq-axi_T2w.nii.gz"}
    if not all(v.is_file() for v in p.values()):
        return None
    img = {k: sitk.ReadImage(str(v), sitk.sitkFloat32) for k, v in p.items()}
    x = {k: N.to_array(v) for k, v in img.items()}
    brain, wm, row = {}, {}, {"subject": sid}
    for k in p:
        brain[k], wm[k], c = N.white_matter(x[k], img[k].GetSpacing())
        row[f"wm_{k}"] = c[0]
        row[f"sigma_lap_{k}"] = N.laplacian_sigma(x[k], wm[k])
    b, corr = N.register(img["HFC"], img["HFE"])
    row["registration_corr"] = corr
    s, d = N.rescan_sigma(x["HFC"], N.to_array(b), wm["HFC"])
    row["sigma_rescan_64mT"] = s
    _, rel = N.noise_spectrum(d, brain["HFC"])          # all of the brain: white matter alone is too patchy
    row["hf_share_64mT"] = N.high_frequency_share(rel)
    row["smoothing_px_64mT"] = N.fit_smoothing(rel)
    row["spectrum_64mT"] = " ".join(f"{v:.3f}" for v in rel)
    shape_fits(row, "3T_lap", *N.laplacian_residuals(x["3T"], wm["3T"]))
    shape_fits(row, "64mT_lap", *N.laplacian_residuals(x["HFC"], wm["HFC"]))
    shape_fits(row, "64mT_rescan", d[wm["HFC"]] / np.sqrt(2), np.nonzero(wm["HFC"])[2])
    info = people.get(sid.removeprefix("sub-"), {})
    row["age"] = float(info.get("age", "nan"))
    row["male"] = float(info.get("sex") == "M")
    row["brain_ml"] = brain["3T"].sum() * np.prod(img["3T"].GetSpacing()) / 1000   # the head > 15 mm deep
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--out", default="results/ds006557_noise.csv")
    ap.add_argument("--subjects", type=int, default=0, help="only the first n (0: all)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    sids = sorted(p.name for p in root.glob("sub-*"))
    sids = sids[:args.subjects] if args.subjects else sids
    with open(root / "participants.tsv") as f:
        people = {r["participant_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    rows = []
    for sid in sids:
        r = subject(root, sid, people)
        if r is None:
            print(f"{sid}: missing a scan, skipped", flush=True)
            continue
        rows.append(r)
        print(f"{sid}: 3T SNR {r['wm_3T'] / r['sigma_lap_3T']:5.1f} (Laplacian) | 64 mT SNR "
              f"{r['wm_HFC'] / r['sigma_rescan_64mT']:4.1f} (rescan), {r['wm_HFC'] / r['sigma_lap_HFC']:4.1f} "
              f"(Laplacian) | smoothing {r['smoothing_px_64mT']:.2f} px | registration r {r['registration_corr']:.3f}",
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows, root)
    shapes(rows)
    controls(rows)
    print(f"wrote {out}")


def summarize(rows, root):
    q = lambda f: np.percentile([f(r) for r in rows], [25, 50, 75])
    fmt = lambda v, d=1: f"{v[1]:.{d}f} [{v[0]:.{d}f}-{v[2]:.{d}f}]"
    snr3 = q(lambda r: r["wm_3T"] / r["sigma_lap_3T"])
    rescan = q(lambda r: r["wm_HFC"] / r["sigma_rescan_64mT"])
    lap = q(lambda r: r["wm_HFC"] / r["sigma_lap_HFC"])
    rep = q(lambda r: r["sigma_lap_HFE"] / r["wm_HFE"] / (r["sigma_lap_HFC"] / r["wm_HFC"]))
    print(f"\n{len(rows)} subjects, white matter, median [quartiles]")
    print(f"  3 T    SNR {fmt(snr3)} per 1 mm^3 voxel (Laplacian, assumes white noise); "
          f"{snr3[1] * np.sqrt(LOW_VOXEL_MM3):.0f} scaled to a {LOW_VOXEL_MM3:.2f} mm^3 voxel")
    print(f"  64 mT  SNR {fmt(rescan)} from scan-rescan, {fmt(lap)} from the Laplacian: the true value lies between")
    print(f"  64 mT  in-plane smoothing of the noise {fmt(q(lambda r: r['smoothing_px_64mT']), 2)} px (Gaussian SD; "
          f"0 = white), high-frequency share {fmt(q(lambda r: r['hf_share_64mT']), 2)} (white = 1)")
    print(f"  64 mT  scanner E / scanner C noise (Laplacian, relative to WM) {fmt(rep, 2)}")
    t64 = json.load(open(next(root.glob("sub-*/ses-HFC/anat/*acq-axi_T2w.json")))).get("AcquisitionDuration")
    if t64:
        print(f"  64 mT  scan time {t64 / 60:.1f} min (2 averages)")


def shapes(rows):
    """Per sample set: each family's held-out surprisal against the Gaussian's, its fitted SD and how often it wins."""
    for name, what in (("3T_lap", "3 T, Laplacian residuals"), ("64mT_lap", "64 mT, Laplacian residuals"),
                       ("64mT_rescan", "64 mT, rescan difference / sqrt(2)")):
        bits = {f: np.array([r[f"bits_{name}_{f}"] for r in rows]) for f in Dist.FAMILIES}
        wins = np.argmin(np.stack(list(bits.values())), 0)
        print(f"\n  {what}: held-out surprisal against the Gaussian (bits/sample, negative = better), "
              f"SD at the minimum (robust-SD units), subjects where it is best")
        for i, (f, b) in sorted(enumerate(bits.items()), key=lambda t: np.median(t[1][1] - bits["Gaussian"])):
            sd = np.median([r[f"sd_{name}_{f}"] for r in rows])
            print(f"    {f:22s} {np.median(b - bits['Gaussian']):+7.4f}   SD {sd:4.2f}   best in {np.sum(wins == i):2d}")


def ols(y, cols, names, n_boot=2000, seed=0):
    """Least-squares coefficients with bootstrap (over subjects) 95 % intervals."""
    X = np.column_stack([np.ones(len(y))] + cols)
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    X, y = X[ok], y[ok]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(len(y), size=len(y))
        boot.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5], 0)
    return {n: (b, l, h) for n, b, l, h in zip(["intercept"] + names, beta, lo, hi)}


def controls(rows):
    """Noise relative to white matter against the person (age, sex, head size), and scanner E against scanner C."""
    from scipy import stats
    g = lambda k: np.array([r[k] for r in rows], float)
    cols = [(g("age") - 45) / 10, g("male"), (g("brain_ml") - np.median(g("brain_ml"))) / 100]
    names = ["age (per decade)", "male", "brain volume (per 100 ml)"]
    print(f"\n  noise / white matter (log), against the person: % change per unit, 95 % bootstrap interval "
          f"(n = {len(rows)}; brain volume median {np.median(g('brain_ml')):.0f} ml)")
    for label, y in (("3 T, Laplacian", np.log(g("sigma_lap_3T") / g("wm_3T"))),
                     ("64 mT, rescan", np.log(g("sigma_rescan_64mT") / g("wm_HFC"))),
                     ("64 mT, Laplacian", np.log(g("sigma_lap_HFC") / g("wm_HFC")))):
        fit = ols(y, cols, names)
        print(f"    {label:17s} " + "   ".join(
            f"{n} {100 * (np.exp(b) - 1):+5.1f} % [{100 * (np.exp(lo) - 1):+.1f}, {100 * (np.exp(hi) - 1):+.1f}]"
            for n, (b, lo, hi) in fit.items() if n != "intercept"))
    ratio = g("sigma_lap_HFE") / g("wm_HFE") / (g("sigma_lap_HFC") / g("wm_HFC"))
    p = stats.wilcoxon(np.log(ratio)).pvalue
    print(f"    scanner E / scanner C (same people, Laplacian noise / WM): median {np.median(ratio):.3f}, "
          f"{np.sum(ratio < 1)} of {len(ratio)} lower, Wilcoxon p = {p:.1g}")


if __name__ == "__main__":
    main()
