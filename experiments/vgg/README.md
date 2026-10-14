# VGG-16 initialization study: minimal extracted path

This directory contains only the CIFAR-10 VGG-16 architecture and five
activations needed to revisit the paper's initialization/learning-rate figure.
It does **not** include the broader private `Musicath-base` or `soft-norm`
framework. `model.py` reproduces the historical 13-convolution layout,
identity normalization (`norm=No`), position-wise PLN[8], fan-out Kaiming
initialization with a multiplicative scale, and the historical `sat` formula.

`train_cifar10.py` is a **reconstructed one-cell runner**, not an archived
per-run result. It uses CIFAR-10 training images with random crop, horizontal
flip, AutoAugment, and the archived channel normalization; the 10,000-image
standard test split is evaluated after each epoch. The historical code names
this loader `val_loader`, but it calls `CIFAR10(..., train=False)`—there is no
separate validation split. The runner uses SGD with momentum 0.9, weight decay
1e-4, batch size 256, a ten-epoch linear warmup, and learning-rate division
by 2.5 before one-based epochs 50/90/130/170 (equivalent to the archived
post-warmup 40/80/120/160 milestones); cuDNN benchmarking is enabled as in the
historical trainer. It writes `config.json` and per-epoch JSONL
records to a new output directory, and never silently replaces existing runs.
The JSONL records both each epoch's `test_accuracy` and the running
`best_test_accuracy`, matching the two quantities observable in the old
trainer. The original plot-generating data were not found, so **do not assume**
which of these quantities produced the manuscript figure. Selecting the best
epoch by test accuracy would also make the test split a selection criterion;
it should not be described as a purely untouched final evaluation in that
case.

Example for **one** PLN cell (not a request to run the whole grid):

Run the command from the repository root using `python -m`; direct execution
of `train_cifar10.py` by file path does not set up the package import path.

```sh
python -m experiments.vgg.train_cifar10 \
  --data-root data/cifar10 --download --output outputs/vgg/pln_lr0.01_init1_seed20 \
  --activation pln --lr 0.01 --init-scale 1 --seed 20 \
  --epochs 200 --device cuda:0
```

The archived private generator spans five activations, seven learning rates,
nine scales and **two** seeds (20 and 30): 630 launch scripts. The current
manuscript describes 945 runs and three seeds. The third seed's launch source
and the per-run results have not been found, so this directory is **not yet**
a verified reproduction of the figure. `--max-train-batches` is only for
development smoke checks and cannot yield paper-comparable accuracy.

For a **new prospective** sweep, `make_grid_plan.py` enumerates the exact five
activations, seven learning rates and nine initialization multipliers stated in
the manuscript. It requires explicit seeds and writes a JSONL plan whose
`argv` arrays invoke the one-cell runner; it does not start training. With
seeds 20 and 30 it produces 630 entries. Adding a third author-chosen seed
produces 945 entries, but does **not** recover the unidentified historical
third seed or its results. For example, if seed 40 is chosen for a new run:

```sh
python -m experiments.vgg.make_grid_plan \
  --seeds 20 30 40 --data-root data/cifar10 \
  --output-root outputs/vgg/runs --plan outputs/vgg/grid_plan.jsonl
```

The plan and future run directories are under ignored `outputs/`. Review the
plan, device assignment, data availability and compute budget before executing
any entry. Do not label a newly executed grid as the archived 945-run figure.

After a **new** three-seed, 200-epoch grid has finished, audit every run and
aggregate the fixed-grid cell means with an explicit epoch statistic:

```sh
python -m experiments.vgg.summarize_grid \
  --run-root outputs/vgg/runs --seeds 20 30 40 --metric final \
  --output outputs/vgg/summary_final.csv
```

The summarizer checks all 945 run directories, the saved settings, and all 200
epoch records before writing a 315-cell CSV; missing or shortened runs fail the
audit. `--metric best` is available only as an explicitly labeled alternative.
The original figure's final-versus-best epoch choice is still unverified, so
neither choice is silently presented as a recovered historical result. The
seed `40` here is only an example author-chosen seed for a future replication.

To draw a five-panel heatmap from that verified CSV (not from the lost archive):

```sh
python -m experiments.vgg.plot_grid \
  --summary outputs/vgg/summary_final.csv \
  --output outputs/vgg/vgg_new_replication.pdf
```

The plotter rejects incomplete grids and does not overwrite an existing figure.
It shares a 0--100% color scale across all five nonlinear layers, with learning
rates increasing left to right and initialization scales increasing bottom to top.

On 2026-10-10, an isolated source archive from commit `221c94b` completed
limited CIFAR-10 training on an RTX 4090: one-epoch PLN[8] and ReLU runs with
two training batches each, plus a ten-epoch PLN[8] run retaining the historical
warmup but limiting each epoch to eight training batches. All used seed 20 and
completed without a code-path error. Their logs are identified by SHA-256 in
`../../docs/REPRODUCIBILITY.md`; the resulting accuracies are **not** evidence
for the manuscript's stability figure.

A later 64-file isolated **uncommitted candidate** completed ten uncapped
epochs of the seed-20 PLN[8] cell on RTX 4090 GPU1 with the historical
ten-epoch warmup, batch size 256, learning rate 0.01 and initialization
multiplier 1. All ten epoch records were saved; the final standard CIFAR-10
test accuracy was 65.4%. A separate one-epoch uncapped entry check also
passed. Source and evidence hashes, environment and other family smoke tests
are in `../../docs/GPU_VALIDATION_20261010.md`. Ten epochs do not reproduce
the paper's 200-epoch grid or establish which epoch statistic was plotted.
