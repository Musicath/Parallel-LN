# One full ViT configuration on RTX 4090 (2026-10-10)

This is a single-run check of the standalone arXiv-code candidate, **not** a
reproduction of the manuscript's three-seed ViT grid or its family-wise best
summary. No CIFAR-10 images, checkpoints, or private run logs are released.

## Configuration and source identity

The run used the isolated 65-file candidate archive with SHA-256
`756eccbf5a25704ab17f8872f055ba68928c0920d68bebbfba537459daf8a070`.
It ran on one physical NVIDIA GeForce RTX 4090 (GPU 1) under Python 3.10.16,
PyTorch 2.6.0+cu126, and torchvision 0.21.0+cu126. The historical ViT run
used PyTorch 2.3.0+cu121 and torchvision 0.18.0+cu121, so this is an
environment-sensitive comparison, not a bitwise replay.

The matched historical row is `random20k,allnorm,40,197,48` in
`figures/data/vit_runs.csv`. Both configuration JSONs specify a depth-4,
width-192 ViT, all-normalization TCGCN (`G_T=197`, `G_C=48`), GELU retained,
model seed 40, fixed-label seed 20260629, 20,000 random-label examples,
100 epochs, batch size 256, constant AdamW learning rate `1e-4`, weight decay
`5e-6` with normalization affine excluded, and evaluation every five epochs.
The historical configuration has SHA-256
`4a773e22c2af1aa4eb8607384ad1fb96d02be62a636ddecd1ec216a66fd5459a`.
The regenerated fixed-label artifacts passed the candidate's count and
first-label checks. Prior candidate versions rebuilt identical 6k and 20k
artifacts on this cache; the **original historical 20k artifact digest is
still unavailable**.

## Observed comparison

| Metric (%) | Historical run | Isolated candidate | Difference (candidate minus historical) |
| --- | ---: | ---: | ---: |
| Best random-label fitting accuracy | 82.425 | 81.940 | -0.485 percentage points |
| Final-epoch random-label fitting accuracy | 78.630 | 81.940 | +3.310 percentage points |
| Clean-label train accuracy of the random-label-trained model | 9.765 | 9.655 | -0.110 percentage points |
| Clean-label test accuracy of the random-label-trained model | 9.640 | 9.360 | -0.280 percentage points |

The historical best was at zero-based epoch 94 and the candidate's best was
at zero-based epoch 99. The logs each contain 100 epoch records plus one
final clean-label evaluation record. Intermediate fitting evaluations also
followed a similar trajectory: at epochs 40, 45, and 55 (one-based), the
candidate reported 21.28%, 25.01%, and 35.76%, versus historical 21.665%,
24.915%, and 36.245%. This agreement supports the standalone runner for
**this one configuration**; it does not establish exact numerical identity,
the other groupings, or any three-seed aggregate.

The candidate printed a normal `done` line after epoch 100, and its wrapper
process exited. The complete log, configuration, and metrics were copied to
ignored local `outputs/vit_full_20261010/`; their local SHA-256 values matched
the remote originals:

| Private evidence file | SHA-256 |
| --- | --- |
| `vit_full.log` | `1e50eb7f6042fc51f25cfb0350df0435d0190e338eac118ada5fe209fdd43265` |
| `config.json` | `a0b42193982d1a672dc45d24117b910010e32ac90e85e3f4ae679130c07541fc` |
| `metrics.jsonl` | `b9c478efe9eefe426f045d70c89b3adb00ecd316ad60327801e5840a41b77678` |

The historical per-epoch `metrics.jsonl` has SHA-256
`4e6cb5f4fe7856bc9b46aace0f9873d3008f2068e2817d7cf77c8b7801e2a543`.
Its original log/configuration remain outside this public-release candidate;
only the sanitized numerical row in `figures/data/vit_runs.csv` is included.
