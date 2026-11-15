"""Noise at 64 mT against 3 T in the same people, across datasets, scanners and sessions.

    python scripts/field_noise.py --ds006557 path/to/ds006557 --lumc path/to/lumc/v5/Data [--out results]

For every T2w scan, in white matter (imrecon.noise): its signal, the Laplacian noise SD, and which distribution the
noise follows (imrecon.distributions): Laplacian residuals against the residual families, raw magnitudes of flat
3 x 3 patches against the magnitude families (Gaussian, Rician, noncentral chi), and the air outside the head, where
the vendor left it, against the background families. Every family's surprisal is swept over its SD (or scale) on
even slices and scored at the minimum on odd slices.

For every pair of 64 mT scans of one person (two scanners in ds006557, two sessions on one scanner at LUMC): the
scan-rescan SD, the noise spectrum and its Gaussian smoothing width, and the distribution of the difference.

Then the controls: noise relative to white matter against scanner, brain volume, and (ds006557) age and sex; and
3 T against 64 mT within the same people. Writes scans.csv, pairs.csv and curves.npz (surprisal curves) to --out:
numbers only, no images. scripts/noise_figures.py draws them.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import datasets as DS  # noqa: E402
from imrecon import distributions as Dist  # noqa: E402
from imrecon import noise as N  # noqa: E402

CURVES = {}                                          # (kind, scan id, family) -> surprisal curve on the train half


def load(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    return img, N.to_array(img)


def residual_fits(row, kind, sid, samples, slices):
    v = (samples - np.median(samples)) / N.robust_sd(samples)
    train, test = Dist.split(v, slices)
    for fam in Dist.FAMILIES:
        curve = Dist.sweep(train, fam)
        i = int(np.argmin(curve))
        CURVES[(kind, sid, fam)] = curve
        row[f"{kind}|{fam}|bits"] = Dist.surprisal(test, fam, Dist.SD_GRID[i])
        row[f"{kind}|{fam}|sd"] = Dist.SD_GRID[i]


def scan_row(s):
    img, x = load(s["path"])
    vox = img.GetSpacing()
    brain, wm, c = N.white_matter(x, vox)
    sid = f"{s['dataset']}/{s['subject']}/{s['session']}"
    row = {k: s[k] for k in ("dataset", "subject", "session", "field_T", "scanner", "age", "sex")}
    row.update(id=sid, voxel_mm3=float(np.prod(vox)), wm=c[0], brain_ml=brain.sum() * np.prod(vox) / 1000)
    row["sigma_lap"] = N.laplacian_sigma(x, wm)
    residual_fits(row, "residual", sid, *N.laplacian_residuals(x, wm))

    patches, sl = N.flat_patches(x, wm)                 # magnitude families, in units of the patch noise SD
    if len(patches) > 200:
        s0 = N.robust_sd((patches - patches.mean(1, keepdims=True)).ravel()) * np.sqrt(9 / 8)
        sig = s0 * np.geomspace(0.3, 3, 121)
        tr, te = patches[sl % 2 == 0], patches[sl % 2 == 1]
        for fam in Dist.MAGNITUDE_FAMILIES:
            curve = np.array([Dist.magnitude_surprisal(tr, fam, g) for g in sig])
            i = int(np.argmin(curve))
            CURVES[("magnitude", sid, fam)] = curve
            row[f"magnitude|{fam}|bits"] = Dist.magnitude_surprisal(te, fam, sig[i])
            row[f"magnitude|{fam}|sd"] = sig[i] / s0
        row["magnitude_snr"] = float(np.median(patches.mean(1)) / s0)

    bg, sl = N.background(x, vox)                       # only where the vendor left real values in the air
    row["background_n"], row["background_median"] = len(bg), float(np.median(bg)) if len(bg) else 0.0
    if len(bg) > 2000 and np.median(bg) >= 5:
        tr, te = Dist.split(bg.astype(float), sl)
        for fam, (sd, bits) in Dist.background_fit(tr, te).items():
            row[f"background|{fam}|bits"], row[f"background|{fam}|sd"] = bits, sd
    return row


def pair_row(a, b):
    from scipy import ndimage as ndi  # noqa: F401
    ia, xa = load(a["path"])
    ib, _ = load(b["path"])
    brain, wm, c = N.white_matter(xa, ia.GetSpacing())
    reg, corr = N.register(ia, ib)
    s, d = N.rescan_sigma(xa, N.to_array(reg), wm)
    _, rel = N.noise_spectrum(d, brain)
    sid = f"{a['dataset']}/{a['subject']}/{a['session']}-{b['session']}"
    row = {"dataset": a["dataset"], "subject": a["subject"], "id": sid, "same_scanner": a["scanner"] == b["scanner"],
           "scanner_a": a["scanner"], "scanner_b": b["scanner"], "wm": c[0], "registration_corr": corr,
           "sigma_rescan": s, "snr_rescan": c[0] / s}
    if rel is not None:
        row["smoothing_px"] = N.fit_smoothing(rel)
        row["hf_share"] = N.high_frequency_share(rel)
        row["spectrum"] = " ".join(f"{v:.4f}" for v in rel)
    residual_fits(row, "rescan", sid, d[wm] / np.sqrt(2), np.nonzero(wm)[2])
    return row


def write(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds006557")
    ap.add_argument("--lumc")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    scans = (DS.ds006557(args.ds006557) if args.ds006557 else []) + (DS.lumc(args.lumc) if args.lumc else [])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(scans):
        rows.append(scan_row(s))
        r = rows[-1]
        print(f"[{i + 1}/{len(scans)}] {r['id']:28s} {r['field_T']:5.3f} T  SNR {r['wm'] / r['sigma_lap']:5.1f} "
              f"(Laplacian)", flush=True)
    write(rows, out / "scans.csv")
    pairs = []
    for a, b in DS.rescan_pairs(scans):
        pairs.append(pair_row(a, b))
        p = pairs[-1]
        print(f"pair {p['id']:32s} SNR {p['snr_rescan']:5.1f} (rescan)  smoothing {p.get('smoothing_px', np.nan):.2f} px",
              flush=True)
    write(pairs, out / "pairs.csv")
    np.savez(out / "curves.npz", keys=np.array(["|".join(k) for k in CURVES]),
             curves=np.array(list(CURVES.values()), dtype=object), sd_grid=Dist.SD_GRID, allow_pickle=True)
    print(f"wrote {out}/scans.csv ({len(rows)}), pairs.csv ({len(pairs)}), curves.npz")


if __name__ == "__main__":
    main()
