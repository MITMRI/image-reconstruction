# image-reconstruction

Tools for reconstructed MR images. First piece: how noise differs between low and high field in the same people, a
noise model a simulator can draw from, and a check of that model by turning 3 T scans into 64 mT ones.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/field_noise.py --ds006557 path/to/ds006557 --lumc path/to/lumc/v5/Data   # ~10 min
.venv/bin/python scripts/noise_report.py results          # tables and figures 1-6
.venv/bin/python scripts/noise_examples.py --ds006557 path/to/ds006557 --lumc path/to/lumc/v5/Data   # 7, 8
cd scripts && ../.venv/bin/python degrade_3t.py path/to/ds006557 --out ../results            # 9, 10
cd scripts && ../.venv/bin/python fake_panel.py path/to/ds006557 --out ../results/figures      # 11
```

| module | what it does |
|---|---|
| `imrecon.datasets` | T2w scans of both datasets as one list (field, scanner serial, session; age and sex where published) |
| `imrecon.noise` | head and white-matter masks; noise SD from the Laplacian stencil (one scan) or a registered rescan (two scans); noise spectrum and the smoothing width that fits it; flat patches and background samples |
| `imrecon.distributions` | which distribution the noise follows: sweep each family's SD, measure surprisal (bits/sample), keep the minimum, compare families on held-out slices. Residual families (Gaussian, Laplace, logistic, Student-t, generalized normal, hyperbolic secant, Cauchy, skew-normal, NIG, Johnson SU), magnitude families (Gaussian, Rician, noncentral chi for 2-8 coils) and background families (Rayleigh, chi, Maxwell, half-normal, exponential, gamma, lognormal) |
| `imrecon.degrade` | a 3 T scan made 64 mT-like in steps: resampled, + resolution, + contrast (intensity map, physics, tissue signals from another 64 mT scan, optionally with T1w tissue labels), + noise |

Data stay in a local cache; scripts write numbers to `results/` (not committed). Figures 7 and 9 show a person's
scans and must never be committed or shared outside the team.

## Data

- [OpenNeuro ds006557](https://doi.org/10.18112/openneuro.ds006557.v1.0.2) (Váša et al. 2025, CC0): 23 adults,
  GE SIGNA Premier 3 T CUBE T2w (1 mm) and axial T2w (1.5 × 1.5 × 5 mm, 6.4 min) on two Hyperfine Swoop scanners
  (serials …41 and …45) days apart. The 64 mT files list `MagneticFieldStrength` 0.006 T; their imaging frequency,
  2.71 MHz, is 64 mT.
- [LUMC paired 64 mT / 3 T, Zenodo 20281403 v5](https://zenodo.org/records/20281403) (van den Broek, Webb, Lena; MIT
  licence): axial T2w (1.6 × 1.6 × 5 mm, 2.6 min) on one Hyperfine Swoop (…44), 1–5 sessions for 66 people, and a
  Philips Achieva 3 T T2w at matched resolution (1.4 × 1.4 × 5 mm) for 11 of them. One 64 mT scan, reconstructed at
  0.8 mm, is left out (interpolated pixels share their noise). **The 3 T JSON sidecars still contain a patient ID,
  accession number, birth date, age and sex** despite the README saying identifiers were removed; this code never
  reads them. Worth telling the authors.
- Not used: [MRI-PairedField](https://github.com/PeirongLiL/MRI-PairedField). Its 64 mT images are these two datasets,
  resampled to 0.5 mm, normalized and with the background masked, which removes what a noise analysis needs.

## Method

Vendors leave little usable background (GE quantizes the air to 0 or 1; Hyperfine masks much of it), so noise is
measured in white matter: with Immerkær's Laplacian stencil (exact for white noise, blind to smoothed noise), and
from rescans (register, match intensities, robust SD of the difference / √2: sees all frequencies, plus
registration error). The two bracket the truth. The rescan difference's spectrum gives the in-plane Gaussian width
that turns white noise into the measured noise. For the distribution, each family's SD (or scale) is swept, the
surprisal minimum taken on even slices and scored on odd slices. Controls: least squares of log(noise / white matter)
on scanner, brain volume and (ds006557) age and sex, with bootstrap intervals.

## Results

**SNR in white matter** (Laplacian, median):

| scanner | voxel | SNR |
|---|---|---|
| GE 3 T (ds006557) | 1 mm³ | 42 |
| Philips 3 T (LUMC) | 9.45 mm³ | 180 |
| Hyperfine …41 / …45 (ds006557) | 11.25 mm³ | 18 / 20 |
| Hyperfine …44 (LUMC) | 12.8 mm³ | 13.5 |
| 64 mT from rescans (ds006557 / LUMC) | | 8.2 / 8.0 |

The Laplacian sees only about 45 % of the 64 mT noise, because that noise is smoothed (by 0.5–0.6 px in-plane):
real 64 mT SNR lies between the rescan and Laplacian figures. In the same LUMC people at matched resolution, 3 T has
about 16 times the SNR per √mm³ by the Laplacian (more by the rescan figure).

**Distribution.** 64 mT noise is close to Gaussian: the best family (Student-t, ν = 10–30) beats the Gaussian by
only 0.002–0.006 bits/sample. In magnitude, Rician and noncentral chi are indistinguishable from Gaussian at these
SNRs (≤ 0.003 bits), as theory says. 3 T residuals are heavy-tailed (Student-t, ν ≈ 4, 0.11 bits better), which
one scan cannot separate from fine anatomy or noise that varies in space (parallel imaging). In the air around 64 mT
heads, the textbook Rayleigh loses to a gamma (k = 2) by 0.17 bits/sample: the background holds ghosting and
uneven noise, not just thermal noise.

**Is the noise the same everywhere?** Nearly, in tissue: 13–15 % of white matter across white and grey matter and
the brain centre, but 19 % in bright CSF and 18 % in the top third of the slab (flat voxels; tissue classes from the
3 T scan, whose noise is independent). Thermal noise is uniform in k-space; the rises come from processing
(intensity correction amplifies noise where the coil is weak; denoising depends on signal) and, in CSF, pulsation.

**The scanner matters; the person does not.** Brain volume, age and sex change 64 mT noise / white matter by at
most a few percent (intervals include 0). The Hyperfine units differ by 24–32 % (different protocols across the two
datasets), and by 10 % in the same people on the same protocol (…45 vs …41, all 23 people, p = 2 × 10⁻⁷).

**Noise model to simulate Hyperfine T2w:** Gaussian noise, blurred in-plane by 0.6 px, SD = white matter / 8
(rescan) to / 18 (Laplacian), × about 1.2 toward the top of the slab and 1.3 in CSF, ± 10–30 % per scanner;
gamma-like air where it is not masked.

## 3 T to 64 mT (`scripts/degrade_3t.py`, `scripts/fake_panel.py`; 23 people)

Each person's 3 T scan is degraded onto scanner C's grid and compared with scanner C's scan, which the degrader never
sees; anything learned comes from scanner E's scan of the same person. Yardstick: real minus rescan (two noises plus
registration). Ratio = SD(real − fake) / SD(real − rescan); 1 means as close as a rescan.

| step | white matter | whole brain | brightest quarter: real − fake |
|---|---|---|---|
| 0 resampled | 1.21 | 1.95 | +152 % of WM |
| 1 + resolution (in-plane Gaussian, 5 mm slab box) | 1.16 | 1.84 | +156 % |
| 2a + contrast: intensity map (paired class means) | 1.15 | 1.70 | +125 % |
| 2b + contrast: physics, 50 mT table | 1.08 | 1.83 | +164 % |
| 2c + contrast: physics, tissue signals from scanner E | 1.28 | 1.76 | +117 % |
| 2d + contrast: as 2c, tissue labels from the 3 T T1w | 1.37 | 1.66 | +108 % |
| 3 + noise model (Student-t ν = 10, on 2d) | 1.54 | 1.75 | +108 % |

- **The noise matches**: the best-fit family (Student-t, ν = 10), SD from the rescans, smoothed 0.6 px. The fake's
  Laplacian SNR is 20 against the real 18, its spectrum smoothed by 0.70 px against 0.58.
- **Physics contrast**: tissue fractions from the 1 mm 3 T scan, a 64 mT spin-echo signal per tissue (TR 2 s, TE
  161 ms), then the blur, so partial volume mixes 64 mT signals. With the 50 mT literature T1/T2 it gives grey matter
  1.19 and CSF 2.61 times white matter; the person's other 64 mT scan gives 1.74 and 4.77 (class means; a
  least-squares fit of the blurred fraction maps could not tell grey from white matter and gave 1.0). The scalp is
  its own tissue (0.91 times white matter), because the 3 T CUBE scan is fat-suppressed.
- **Deep grey matter, and why a T1w helps**: iron in the basal ganglia shortens T2 more at high field, so those
  nuclei are dark at 3 T and not at 64 mT, and tissue classes from 3 T T2w intensity call them white-matter-like (fake
  27 % too dark there in one person). Tissue labels from the 3 T T1w (same session; resampled in scanner coordinates,
  not registered) fix much of it: the fake's correlation with the real scan in the brain rises from 0.60 to 0.65
  (rescan 0.94) and the whole-brain ratio from 1.76 to 1.66, though white matter gets worse (1.28 to 1.37). The T1w
  only labels tissue; the output is still a 64 mT T2w. For a pipeline from one high-field scan of any contrast, the
  labels should come from a contrast-agnostic segmenter (SynthSeg) instead. Shifting the fake along the slab confirmed
  the 3 T to 64 mT registration (best match at 0 to +2 mm).
- **What is still missing, in the brightest tissue** (CSF and cortex): the fake is far too dark there in every
  version. Less blur makes it worse, not better (brain ratio 1.80 → 1.93 from a full to no in-plane blur), so it is
  not the blur. Likely causes, visible in `11_fake_panel.png`: the 3 T CUBE scan is fat-suppressed (its scalp is
  dark; 64 mT's is bright), sulcal CSF is far brighter at 64 mT than three tissue classes from 3 T predict, and the
  64 mT ventricles look wider than the 3 T ones in the same 5 mm slab.
- Contrast maps tried and dropped: a voxel-wise lookup (pulled toward the mean), quantile matching (copies the
  noise-widened 64 mT histogram), and class means paired by rank (a vessel class at 3 T shifts the pairing).

## Limits

- Vendor-reconstructed, denoised images: this is the noise of product images, not of the hardware.
- Field strength is confounded with vendor, coil, sequence and reconstruction; the 3 T / 64 mT ratio is not a
  field-strength law.
- Rescan differences include registration error and different 5 mm slice placement.
