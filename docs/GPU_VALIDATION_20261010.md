# Isolated RTX 4090 candidate validation (2026-10-10)

This records **execution checks**, not reproduction of the manuscript's
multi-seed statistics. The author-approved validation used physical RTX 4090
GPU 1 on the eight-GPU host after it was observed idle. CIFAR-10 was read from
an existing local cache and was not copied into this repository.

The candidate was assembled from the 64 tracked or non-ignored untracked files
in the working tree at test time. Its source archive SHA-256 was
`e5304aea589e38073f3e7f351643f354a5a1de643c1a18e91fc8f977cb9d8525`.
It was extracted to a private temporary directory, with the archive digest
checked after transfer. These files included uncommitted changes and are **not**
identified by the repository's current Git commit. The existing host
environment was Python 3.10.16, PyTorch 2.6.0+cu126, and torchvision
0.21.0+cu126; it was not freshly installed from the release requirements.
All commands ran from the extracted repository root with `PYTHONPATH` cleared
and `CUDA_VISIBLE_DEVICES=1` (thus `cuda:0` inside the process).

| Training path | Bounded configuration | Observed result | Interpretation |
| --- | --- | --- | --- |
| VGG-16 / PLN[8] | CIFAR-10 full training split, batch 256, seed 20, initialization multiplier 1, SGD learning rate 0.01, ten **uncapped** epochs with the historical ten-epoch linear warmup | Ten epoch records; final test accuracy 65.4% | Standalone runner, data augmentation, scheduler, and evaluation path execute. This is not the paper's 200-epoch/945-run stability figure. |
| VGG-16 / PLN[8] | Same cell, one uncapped epoch with warmup shortened to one epoch | One epoch record; test accuracy 8.98% | Entry-path check only; the changed warmup prevents paper comparison. |
| CNN / PLN[4] | Built-in `--smoke`: 256 fixed random labels, depth 2, width 128, learning rate `1e-3`, one epoch | Final fitting accuracy 19.5312% | Training/data/serialization check, not the paper's full CNN grid. |
| Dense Fourier / PLN[4] | Dimension 4, 8 modes, support 4, width 1024, depth 1, learning rate `1e-4`, 100 warmup-cosine steps, 1,000/200/1,000 train/validation/test samples | Finite test relative L2 `0.9903122` | Short optimization check; the paper uses 10,000 steps and larger datasets. |
| ViT / position-wise PLN | Rebuilt exact first-6k fixed-label artifact; depth 4, width 192, token groups 197, channel groups 48, seed 40, one epoch | Random-label evaluation 11.68%; clean-test accuracy 11.16% | Artifact, training and evaluation paths execute; no claim about the paper's 100-epoch/three-seed grid. |

The same source snapshot passed `tools/validate_candidate.py`: 43 tests in
normal mode and 43 in optimized (`-O`) mode, all three data verifiers in both
modes, ten entrypoint `--help` checks, two PINN dry runs, and generation of
eight nonempty figures. This was on an existing environment, not a fresh
dependency installation or hosted CI.

New logs, configurations and numeric outputs were copied into local **ignored**
`outputs/candidate_gpu1_20261010_evidence/`. The four-family short-run evidence
archive SHA-256 is
`a61ae999183c67f8c95c4dab7e133858fe516e8c6c29801a391ea229577b829e`;
the separate ten-epoch VGG evidence archive SHA-256 is
`2cc1a773daf66bd8a8880abf23f8a2250402fe9cbff6da59f8a8a7c83c624703`.
The validator log SHA-256 is
`9c060fc26340fa4af5c62893640fe6fcb2786dec29378ba18b9fabc8319a107b`;
the ten-epoch VGG metrics JSONL SHA-256 is
`b490415827fb7ac7a2f193c73f408c5c0d4987616ae70d79bea03cf368c83f71`.
These private evidence files can contain local paths and must be reviewed before
any selective publication. The CIFAR images, fixed-label `.pt` artifacts,
historical private framework, and raw W&B logs are not part of this archive.

Full representative configurations for Dense Fourier, CNN, and ViT were run
subsequently; see the separate full-run reports in `docs/`. This earlier smoke
check still does not resolve the missing historical VGG figure data, prove
third-party source rights, or replace the final staged-file review. The author
lifted the earlier publication hold only after these later checks were reported.
