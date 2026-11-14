# image-reconstruction

Tools for reconstructed MR images. First piece: how noise differs between low and high field in the same people, and
the parameters of a noise model a simulator can draw from.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/paired_field_noise.py path/to/ds006557      # about 1 min; CSV to results/ (not committed)
```

| module | what it does |
|---|---|
| `imrecon.noise` | head and white-matter masks; noise SD from the Laplacian stencil (one scan) or from a registered rescan (two scans); the in-plane noise spectrum and the Gaussian smoothing width that fits it |
| `imrecon.distributions` | which distribution the noise follows: sweep each family's SD, measure surprisal (bits/sample), keep the minimum, compare families on held-out slices |
| `scripts/paired_field_noise.py` | all of it on OpenNeuro ds006557, plus age, sex and brain volume as controls |

No scans or images of people go in this repo; data stay in a local cache and the script writes numbers only.

## Data

[OpenNeuro ds006557](https://doi.org/10.18112/openneuro.ds006557.v1.0.2) (Váša et al. 2025, CC0): 23 healthy adults
aged 20–69, each scanned at 3 T (GE SIGNA Premier, 48-channel coil, CUBE T2w, 1 mm isotropic) and at 64 mT on two
identical Hyperfine Swoop scanners days apart (axial T2w, 1.5 × 1.5 × 5 mm, 2 averages, 6.4 min, "advanced
reconstruction"). Only the T2w scans were used (3 T and both 64 mT axial; 268 MB). The 64 mT files list
`MagneticFieldStrength` as 0.006 T; their imaging frequency, 2.71 MHz, is 64 mT.

## Method

Neither vendor leaves usable background: GE quantizes the air around the head to 0 or 1, and Hyperfine masks most of
it. So noise is measured inside the brain, in white matter (the darkest of three intensity classes more than 15 mm
inside the head), three ways:

- **Laplacian** (Immerkær): a 3 × 3 second-difference stencil on each in-plane slice over the flattest half of white
  matter. Exact for white noise, but it only sees high spatial frequencies, so it misses smoothed noise.
- **Scan-rescan** (64 mT only): register the second 64 mT scan onto the first (rigid, correlation), match intensities,
  and take the robust SD of the difference / √2. It sees noise at all frequencies, plus what registration and
  different 5 mm slice placement leave behind, so it is an upper bound; the Laplacian is a lower bound.
- **Spectrum**: the in-plane power spectrum of the rescan difference over the brain, and the in-plane Gaussian width
  that turns white noise into that spectrum.

Distribution shape: Laplacian residuals (both fields) and the rescan difference (64 mT), each centred and divided by
its robust SD. For each family (Gaussian, Laplace, logistic, Student-t with ν = 3–30, generalized normal with β =
0.75–4), the SD is swept from 0.3 to 3 robust SDs and the mean surprisal −log₂ p(x) computed at each. The minimum is the
family's maximum-likelihood SD. Families are compared by surprisal at that SD on the other half of the slices
(fitted on even slices, scored on odd).

Controls: least squares of log(noise / white matter) on age, sex and brain volume, with bootstrap intervals over
subjects; the two 64 mT scanners are compared within each person.

## Results (23 subjects; median [quartiles])

| | 3 T | 64 mT |
|---|---|---|
| Voxel | 1 mm³ | 11.25 mm³ |
| White-matter SNR | 39.8 [36.4–45.1] (Laplacian) | between 8.4 [8.0–8.8] (rescan) and 18.8 [18.3–19.9] (Laplacian) |
| Same, per 11.25 mm³ voxel | ≈ 133 if the noise is white | 8.4–18.8 |
| Noise texture | not measurable from one scan | smoothed: Gaussian width 0.59 [0.57–0.62] px in-plane; power above 0.3 cycles/px is 0.12 of white |
| Best distribution (held-out surprisal) | Student-t, ν ≈ 4: 0.10 bits/sample better than Gaussian, 20 of 23 people | Student-t, ν ≈ 10: only 0.016 bits/sample better than Gaussian (17 of 23); Gaussian is close |
| Maximum-likelihood SD (Gaussian) | 1.28 robust SDs | 1.06–1.08 robust SDs |

**The 64 mT noise is smoothed and close to Gaussian.** Its spectrum falls off as if white noise had been blurred by
a 0.6-pixel Gaussian, consistent with a denoising reconstruction. That is why the Laplacian sees only about 45 % of it
(SNR 18.8 against 8.4). The true SNR lies between the two; the rescan figure also contains registration error, so it
is the pessimistic end. Nearly Gaussian means the extra tails are worth only 0.016 bits per sample.

**The 3 T residuals are heavy-tailed** (Student-t, ν ≈ 4), unlike the 64 mT ones. With one scan this cannot be told
apart from fine anatomy at 1 mm getting into the residual, or from noise that varies across the head (parallel
imaging at R = 3 makes the noise level vary in space, and a mix of Gaussians of different SDs is heavy-tailed).
So the 3 T SNR is approximate.

**The person barely matters at 64 mT; the scanner does.**

| noise / white matter, % change per | age (per decade) | male | brain volume (per 100 ml) |
|---|---|---|---|
| 3 T, Laplacian | +9.7 [+7.5, +12.2] | +0.9 [−5.9, +8.7] | +3.4 [0.0, +6.3] |
| 64 mT, rescan | +2.1 [+0.6, +3.8] | +5.9 [−1.2, +14.5] | +0.5 [−2.2, +2.9] |
| 64 mT, Laplacian | −1.0 [−2.8, +1.3] | −0.1 [−6.9, +6.8] | +3.3 [+0.4, +6.9] |

The second 64 mT scanner (E) reads 9 % less noise relative to white matter than the first (C) in all 23 people
(median ratio 0.907, Wilcoxon p = 2 × 10⁻⁷): identical model and software, yet the scanner differs more than age,
sex or brain size do. It is confounded with session (C was the same day as the 3 T scan, after 45 minutes in it).
The 3 T age effect is most likely white matter itself changing with age, not noise.

## A noise model to simulate, per scan

| parameter | 64 mT Hyperfine, T2w | 3 T GE, T2w |
|---|---|---|
| SNR in white matter | 8–19 (11.25 mm³, 6.4 min) | ≈ 40 (1 mm³) |
| spatial texture | white noise blurred by a 0.6 px in-plane Gaussian | unknown (one scan) |
| distribution | Gaussian (Student-t ν = 10 at most) | heavy-tailed (Student-t ν ≈ 4), cause unclear |
| scanner-to-scanner | ±5–10 % in SD | — |

To simulate a Hyperfine-like image: add Gaussian noise, blur it in-plane with a 0.6-pixel Gaussian, and scale it to
the target SNR. Raw-data reconstructions (such as Leiden's 47 mT system) do not denoise, so their noise should be
close to white.

## Limits

- One dataset, one contrast, one vendor per field. Field strength is confounded with vendor, coil, sequence and
  reconstruction, so the 3 T to 64 mT ratio is not a field-strength law.
- 64 mT images are vendor-reconstructed and denoised; this is the noise of the product images, not of the hardware.
- The rescan SD includes residual misregistration and different slice placement (5 mm slices). B-spline resampling of
  the second scan smooths its noise slightly the other way.
- The LUMC paired 64 mT / 3 T set (Zenodo 20281403) has same-scanner repeats and would separate scanner from session;
  not yet used.
