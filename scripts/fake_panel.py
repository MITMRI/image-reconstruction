"""One person, one slice: 3 T, real 64 mT, 3 T brought down to 64 mT with the noise model, and real minus fake.

    python scripts/fake_panel.py path/to/ds006557 [--subject sub-HYPE01] [--out results/figures]

The fake is imrecon.degrade's last stage (resolution; contrast from the person's other 64 mT scan, with SynthSeg or
T1w tissue labels when given; noise). A
fifth panel shows real minus rescan, the size a perfect fake's difference would have. Writes 11_fake_panel.png; it
shows a person's scans, so it is never committed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imrecon import datasets as DS  # noqa: E402
from imrecon import degrade as G  # noqa: E402
from imrecon import noise as N  # noqa: E402


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    from noise_report import style
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--subject", default="sub-HYPE01")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--synthseg", help="folder of mri_synthseg label maps for the 3 T T2w scans")
    args = ap.parse_args(argv)
    by = {s["session"]: s for s in DS.ds006557(args.root) if s["subject"] == args.subject}
    c, e, g = (sitk.ReadImage(str(by[k]["path"]), sitk.sitkFloat32) for k in ("HFC", "HFE", "GE"))
    real = N.to_array(c)
    brain, wm, _ = N.white_matter(real, c.GetSpacing())
    t1 = Path(str(by["GE"]["path"]).replace("_ses-GE_T2w", "_ses-GE_acq-MPRAGE_T1w"))
    seg = Path(args.synthseg) / Path(str(by["GE"]["path"])).name.replace("_T2w.nii.gz", "_synthseg.nii.gz") \
        if args.synthseg else None
    fakes, _, _ = G.stages(g, c, e, wm, seed=0, t1=sitk.ReadImage(str(t1), sitk.sitkFloat32) if t1.is_file() else None,
                           seg=sitk.ReadImage(str(seg)) if seg and seg.is_file() else None)
    fake = fakes[G.STAGES[-1]]
    three = fakes[G.STAGES[0]]                                   # 3 T on the 64 mT grid, no blur
    t, _ = G.register_transform(c, e)
    rescan = G.resample(e, c, t)
    rescan *= real[wm].mean() / rescan[wm].mean()
    z = real.shape[2] // 2
    vmax = np.percentile(real[wm], 99) * 3
    lim = 4 * N.robust_sd((real - rescan)[wm])
    style(plt)
    panels = [(three, "1  3 T (on the 64 mT grid)", "gray", 0, vmax),
              (real, "2  64 mT, real", "gray", 0, vmax),
              (fake, "3  3 T brought down to 64 mT", "gray", 0, vmax),
              (real - fake, "4  2 - 3: real - fake", "RdBu_r", -lim, lim),
              (real - rescan, "5  for scale: real - rescan", "RdBu_r", -lim, lim)]
    fig, ax = plt.subplots(1, 5, figsize=(18, 4.5))
    for a, (im, title, cmap, lo, hi) in zip(ax, panels):
        h = a.imshow(np.rot90(im[:, :, z]), cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(title, fontsize=9, loc="left")
        a.axis("off")
    fig.colorbar(h, ax=ax[3:].tolist(), fraction=0.02, pad=0.01).ax.tick_params(labelsize=7)
    sd = lambda d, m: N.robust_sd(d[m]) / real[wm].mean() * 100
    fig.suptitle(f"{args.subject}, slice {z}. SD in white matter / brain, % of white matter: real - fake "
                 f"{sd(real - fake, wm):.0f} / {sd(real - fake, brain):.0f}, real - rescan {sd(real - rescan, wm):.0f} / "
                 f"{sd(real - rescan, brain):.0f}", x=0.01, ha="left", fontsize=10, fontweight="bold")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "11_fake_panel.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out / '11_fake_panel.png'}")


if __name__ == "__main__":
    main()
