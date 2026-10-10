# Reproducibility status and commands

This document separates **archived numerical evidence**, **available training code**, and **fresh re-runs**. A plotted CSV or a source file alone is not proof that a historical experiment has been re-run. Status below refers to the expanded arXiv manuscript as inspected during this code release preparation.

| Manuscript item | Training source | Bundled numerical data | Status |
| --- | --- | --- | --- |
| Dense Fourier MLP table, dimensions 4 and 8 | `experiments/dense_fourier/run_dense_fourier_mlp.py` | `figures/data/dense_fourier_d{4,8}_runs.csv` (720 runs per dimension) and matching table summaries | Historical run grid and manuscript aggregation verified; fresh re-run pending |
| Helmholtz PINN tables | Paused pending the authors' original logs or scripts | Manuscript values only; original run logs not located | **Not independently reproducible from this release yet**; do not substitute the later k=1/k=4 PINN protocol |
| Mixed-axis CNN table and figures | `experiments/cnn/run_cifar10_conv_gn_{random,clean}_labels.py` | `figures/data/cnn_runs.csv` (288 per-seed records) | Data completeness and paper checkpoints pass; fresh GPU re-run pending |
| VGG-16 initialization figure | Generic `Musicath-base` VGG backbone located; matching 945-run sweep not located | Verified plotting data not located | Not yet included as reproducible code/data |
| ViT table and grouping figures | Extracted `experiments/vit/model.py` and adapted `train_vit_token_grouping.py` | `figures/data/vit_runs.csv` (1,377 per-seed records) plus exact 6k labels | Historical model-equivalence and data checks pass; fresh GPU re-run pending |

The machine-translation experiments remain outside the active manuscript and are not part of this release.

## Local checks without training

```sh
python -m unittest discover -s tests -v
python figures/verify_data.py
python figures/verify_fourier_data.py
python figures/plot_from_csv.py
python figures/plot_normalization_domain_schematic.py
```

`verify_data.py` checks CNN/ViT row completeness and selected manuscript-level checkpoints; it does not audit every original training log. `verify_fourier_data.py` independently reconstructs the shared-learning-rate selection and two Dense Fourier table summaries from all 1,440 sanitized per-run records. The plotting scripts write to the ignored `figures/generated/` directory. These figures use compact per-seed CSVs and may differ typographically from manuscript PDF assets.

The [ViT-specific guide](../experiments/vit/README.md) covers fixed 6k/20k
artifact generation, the extracted model and one-cell training commands. Its
6k label manifest was checked against an archived run's label counts and first
16 labels. The 20k extension follows the archived preparation algorithm and,
when generated on the eight-RTX-4090 host using an existing CIFAR-10 cache,
matched the archived 20k run's label counts, changed-label count and first 16
clean/random labels. The original 20k artifact digest is not yet available for
bytewise comparison.

## Training entry points already included

The scripts below are the source files used for the corresponding research runs, with machine-independent input/output arguments. They have **not yet** passed fresh GPU verification in this repository. Keep results outside the repository or under the ignored `outputs/` directory.

The Dense Fourier runner generates a fixed target from `--target-seed` and splits from `--data-seed`. The paper sweeps target seeds `20260903, 20260913, 20260923, 20260933, 20260943`, model seeds `0, 1, 2`, learning rates `1e-4, 3e-4, 1e-3`, depths `1, 4`, and the eight listed nonlinearities for each of dimensions 4 and 8. Every run uses 500 warmup steps followed by cosine decay. A single configuration can be launched as follows; it represents one of the 1,440 archived cells, not the full sweep:

```sh
python experiments/dense_fourier/run_dense_fourier_mlp.py \
  --output outputs/fourier --device cpu --input-dim 4 --modes 8 \
  --support-size 4 --width 1024 --depth 1 --activation pln4 --affine \
  --lr 0.001 --lr-schedule warmup-cosine --warmup-steps 500 \
  --steps 10000 --batch-size 256 \
  --n-train 10000 --n-validation 2000 --n-test 10000 \
  --target-seed 20260903 --data-seed 20260904 --model-seed 0 \
  --eval-every 200
```

The CNN grid uses $G_C\in\{1,2,4,8,16,32,64,128\}$ channel groups and $K\in\{1,2,4,8,16,32\}$ spatial splits per axis, with operator names `bgn<G_C>k<K>`. Each cell has model seeds `0,1,2`. These are representative per-cell commands, **not** an instruction to launch all 288 runs at once:

```sh
CUDA_VISIBLE_DEVICES=0 python experiments/cnn/run_cifar10_conv_gn_random_labels.py \
  --data-root data --output outputs/cnn/random --n-train 50000 \
  --channels 128 --depths 4 --pool-size 4 --operators bgn16k32 \
  --lrs 0.01 --model-seeds 0 --data-seed 20260902 --label-seed 20260903 \
  --epochs 200 --batch-size 128 --eval-every 10 \
  --optimizer sgd --momentum 0.9 --weight-decay 0

CUDA_VISIBLE_DEVICES=0 python experiments/cnn/run_cifar10_conv_gn_clean_labels.py \
  --data-root data --output outputs/cnn/clean --n-train 50000 \
  --channels 128 --depth 4 --pool-size 4 --operators bgn16k32 \
  --lr 0.01 --model-seed 0 --data-seed 20260902 \
  --epochs 200 --batch-size 128 --eval-every 10 \
  --optimizer sgd --momentum 0.9 --weight-decay 0
```

These CNN scripts request CIFAR-10 through `torchvision.datasets.CIFAR10` when needed. The training output, downloaded images, and checkpoints are not part of Git. Review the dataset's own terms before redistributing it.

## Environment provenance and GPU validation

Historical observed stacks, **not yet portable lock files**:

| Family | Python | PyTorch | torchvision | NumPy |
| --- | --- | --- | --- | --- |
| CNN/ViT | 3.12.3 | 2.3.0+cu121 | 0.18.0+cu121 | 1.26.4 |
| Dense Fourier | 3.13.2 | 2.6.0+cu124 | Not required | 2.2.4 |
| PINN exploratory runs | 3.10.16 | 2.4.0 | Not required | 2.0.1 |

On 2026-10-10, GPU0 on the eight-RTX-4090 host was idle (1 MiB reported
allocation, 0% utilization); GPU2/3 were busy and untouched. A temporary copy
of the **uncommitted release candidate** was tested with an existing
`pln_vit_iclr2027` environment: Python 3.10.16, PyTorch 2.4.0, torchvision
0.19.0, NumPy 2.2.6, Matplotlib 3.10.3, CUDA 12.1. All eleven unit tests passed;
`figures/verify_data.py` and `figures/verify_fourier_data.py` passed. On GPU0,
a PLN vector forward/backward check had finite gradients, and a depth-4
width-192 ViT all-normalization model produced a finite `(1, 10)` output with
gradients on 56 parameters. Both plotting scripts also completed on Linux with
the non-interactive Matplotlib backend. These are **unit/smoke checks, not
training runs**.
Three CNN tests check that the grid endpoints coincide with conventional GN
and position-wise PLN, and reject invalid block sizes.
Both 6k and 20k fixed-label artifacts were generated in an isolated `/tmp`
directory from the existing CIFAR-10 cache; a second invocation verified them
without overwriting anything.

The historical Dense Fourier environment (`polynomial`) was also inspected
read-only: Python 3.13.2, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.2.4.
Training runners have not been exercised on a clean checkout. Before a full
experiment, record the Git commit, command, output path and GPU state, then run
one short training smoke test and one representative full configuration per
available family. A one-seed run is an execution check, not a verification of
three-seed or five-target paper statistics.
