# ViT token–channel grouping study

This directory contains only the CIFAR-10 ViT path needed for the manuscript.
`model.py` is a minimal extraction of the historical `Musicath-base` backbone,
not a copy of the surrounding framework. At seed 42, its parameter names,
initial values and baseline forward output match the archived backbone exactly;
the two paper replacement modes also have matching parameter names and initial
values. CPU unit tests cover construction and normalization-domain behavior.
These checks do **not** certify that a full training run reproduces the paper.

Run from the repository root with a matching PyTorch/torchvision installation.
Use the `python -m experiments.vit...` form shown below for the package
imports; direct file-path execution is not supported.
The historical environment used Python 3.12.3, PyTorch 2.3.0+cu121,
torchvision 0.18.0+cu121 and NumPy 1.26.4. Matplotlib is needed only for
regenerating plots. SwanLab is optional and disabled unless `--enable-swanlab`
is explicitly passed with `SWANLAB_API_KEY` set and both
`--swanlab-workspace` and `--swanlab-project` specified. No personal
workspace is built into the public runner.
Each `--exp-name` must name a new output directory: the runner refuses to
overwrite or append to an earlier run.

## Fixed labels

The tracked `data/fixed_6k_labels.csv` preserves the exact first-6k CIFAR-10
training indices and clean/random labels. Its random-label SHA-256 tensor digest
is checked by the test suite and its first labels/counts match an archived run
configuration. The historical 20k protocol retains those first 6k labels and
generates the additional 14k from a separate CPU generator seeded 20260629.
On the eight-RTX-4090 host, the generated 20k label counts, changed-label
count, and first 16 clean/random labels matched an archived 20k run config.
The original 20k `.pt` artifact has not been located for a bytewise digest
comparison, so this remains a protocol reconstruction rather than a claim of
byte-for-byte artifact identity.

```sh
python -m experiments.vit.prepare_fixed_labels --size 6000 \
  --data-root data --download --output data/fixed6k
python -m experiments.vit.prepare_fixed_labels --size 20000 \
  --data-root data --output data/fixed20k
```

The builder verifies CIFAR-10's clean labels against the archived 6k manifest
and refuses to overwrite a differing artifact. CIFAR-10 images are downloaded
by torchvision; they are not redistributed in this repository.

## One paper configuration

This command is one all-normalization, random-20k cell. Replace the token and
channel group counts and model seed for other cells. The full grid uses model
seeds 40, 41 and 42; token-group counts
`1,2,4,8,16,32,64,128,197`; channel-group counts
`1,2,4,8,12,24,48,192` for normalization replacement and
`1,2,4,8,12,24,48,96,192` for activation replacement.

```sh
CUDA_VISIBLE_DEVICES=0 python -m experiments.vit.train_vit_token_grouping \
  --data-root data --norm2 TCGCN --norm-scope all --activation-kind gelu \
  --num-tokens 197 --token-groups 32 --channel-groups 192 \
  --depth 4 --width 192 --seed 40 --label-seed 20260629 \
  --data-mode fixed-subset --artifact-dir data/fixed20k --noise-rate 1 \
  --epochs 100 --batch-size 256 --workers 2 \
  --lr 0.0001 --lr-schedule constant --weight-decay 0.000005 \
  --exclude-norm-weight-decay --eval-every 5 --disable-checkpoints \
  --output outputs/vit/random20k --exp-name allnorm_gt32_gc192_seed40 \
  --run-key allnorm_gt32_gc192_seed40
```

For activation replacement, use `--norm2 LN --norm-scope all
--activation-kind tcgcn` and the activation grid's channel-group values.
For clean-label classification, use `--data-mode full-clean --noise-rate 0`
without `--artifact-dir`. The unchanged LN–GELU baseline uses
`--norm2 LN --norm-scope all --activation-kind gelu`.

The archived paper CSVs are in `figures/data/vit_runs.csv`. The training source
was adapted from the random-20k snapshot (SHA-256
`fce28842b43967cbb2ed7b31d2971a95042575cb3a043288091f45da535a9d7b2`):
the external framework import was replaced by the extracted model, data paths
became CLI arguments, CIFAR-10 downloading became opt-in, tensor loading uses
`weights_only=True`, and SwanLab logging became opt-in. Numerical equivalence
of the extracted model was checked locally; fresh GPU training is pending.
