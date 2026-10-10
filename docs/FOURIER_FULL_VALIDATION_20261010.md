# Full-step Dense Fourier representative checks (2026-10-10)

Two **single-target, single-model-seed** PLN[4] configurations were run for
the full 10,000 steps from the isolated 65-file uncommitted candidate archive
(SHA-256 `756eccbf5a25704ab17f8872f055ba68928c0920d68bebbfba537459daf8a070`).
Physical RTX 4090 GPU 1 was idle before use and exposed as `cuda:0` by
`CUDA_VISIBLE_DEVICES=1`. The existing historical Dense Fourier environment
was Python 3.13.2 / PyTorch 2.6.0+cu124; this was **not** the fresh CPU venv
used for the separate release-installation check.

Both runs used 8 Fourier modes, width 1024, depth 1, affine PLN[4], Adam
learning rate `1e-4`, 500 warmup steps followed by cosine decay, batch size
256, 10,000 training points, 2,000 validation points, 10,000 test points,
target seed `20260903`, data seed `20260904`, model seed `0`, and evaluation
every 200 steps. Only input dimension and dense support size changed together:

| Dimension / support | Archived test relative L2 | New test relative L2 | New minus archived | Best step (both) |
| --- | ---: | ---: | ---: | ---: |
| 4 / 4 | `0.23689360916614532` | `0.23691031336784363` | `+1.6704201698303223e-05` | 10,000 |
| 8 / 8 | `0.6083393692970276` | `0.6083347201347351` | `-4.649162292480469e-06` | 9,600 |

The archived values are the matching rows of
`figures/data/dense_fourier_d4_runs.csv` and
`figures/data/dense_fourier_d8_runs.csv`, not newly recovered raw logs. The
relative differences are about `+0.00705%` and `-0.000764%`, respectively.
The new test MSEs were `0.05542948585813052` and `0.37167487764104756`.
These close but nonidentical results provide evidence that two full training
paths behave consistently with the archived rows. They do **not** verify the
five-target/three-seed manuscript aggregates, the learning-rate selection
across those grids, or numerical identity on other machines.

The new logs and JSON results are retained only under ignored local
`outputs/fourier_full_20261010_evidence/`. The evidence archive SHA-256 is
`66434fbf8c25eca3d238a882908377999337962c6499ad7e6aa21800e2dadeb9`;
the four-dimensional and eight-dimensional result JSON SHA-256 values are
`300c5484c8f653f59457e718184624252fb5f30c596b2299c0cc38c371ecb9be`
and `3c1c819e4aa02bcae1a37313b5653e4d96d497bd4c489dfa4b288bd8e4bdaf08`.
The temporary server checkout was removed after the evidence was transferred
and verified locally.
