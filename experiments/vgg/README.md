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
1e-4, batch size 256, a ten-epoch linear warmup, and MultiStepLR milestones
40/80/120/160 with gamma 0.4; cuDNN benchmarking is enabled as in the
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
