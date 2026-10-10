# CIFAR-10 mixed-axis CNN experiments

These two standalone runners implement the depth-4, width-128 CNN study in
the arXiv manuscript. They read CIFAR-10 from a user-provided cache; the
dataset and historical raw training logs are not distributed here. Run the
commands from the repository root with a CUDA-enabled PyTorch installation.
`--download` is opt-in, and both runners refuse to overwrite an existing
per-run JSON result. Their `--smoke` modes change the paper configuration and
are only execution checks.

The operator name `bgn<G>k<K>` splits 128 channels into `G` groups and each
32-pixel spatial axis into `K` contiguous parts. One normalization domain
therefore has shape `(128/G) x (32/K) x (32/K)`. The paper evaluates
`G = 1, 2, 4, 8, 16, 32, 64, 128` and `K = 1, 2, 4, 8, 16, 32` (48 domains),
with model seeds 0, 1, and 2 for each of the random- and clean-label settings.
`bgn1k1` is full-map LN, `bgn64k1` is the two-channel GN endpoint used in the
main-text comparison, and `bgn16k32` / `bgn32k32` are position-wise PLN[8] /
PLN[4]. Operators have no learned affine parameters.

## One paper-protocol random-label cell

The command below selects one of the 144 random-label runs: PLN[4], model
seed 0. It uses the full 50,000-example CIFAR-10 training split with one
fixed permutation of its labels. Training uses SGD, learning rate 0.01,
momentum 0.9, batch size 128, and 200 epochs. There is no augmentation or
weight decay.

```sh
CUDA_VISIBLE_DEVICES=0 python experiments/cnn/run_cifar10_conv_gn_random_labels.py \
  --data-root data --output outputs/cnn/random50k \
  --n-train 50000 --channels 128 --pool-size 4 --depths 4 \
  --operators bgn32k32 --lrs 0.01 --model-seeds 0 \
  --data-seed 20260902 --label-seed 20260903 \
  --epochs 200 --batch-size 128 --optimizer sgd --momentum 0.9 \
  --eval-every 10
```

To reconstruct the domain grid, substitute each `bgn<G>k<K>` combination
and use `--model-seeds 0,1,2`. Separate output directories avoid accidental
mixing with other studies. The runner saves one JSON file per configuration,
including the full evaluation curve, and a `summary.csv`. The manuscript's
random-label figure uses **final** fitting accuracy, not the maximum observed
at an intermediate evaluation.

## One paper-protocol clean-label cell

```sh
CUDA_VISIBLE_DEVICES=0 python experiments/cnn/run_cifar10_conv_gn_clean_labels.py \
  --data-root data --output outputs/cnn/clean50k \
  --n-train 50000 --channels 128 --pool-size 4 --depth 4 \
  --operators bgn32k32 --lr 0.01 --model-seed 0 \
  --data-seed 20260902 --epochs 200 --batch-size 128 \
  --optimizer sgd --momentum 0.9 --eval-every 10
```

Run the clean-label command for each domain and model seed. Its JSON includes
final training and held-out test accuracy, which are distinct from the
random-label fitting metric.

The compact historical per-seed grid is in `figures/data/cnn_runs.csv`.
`python figures/verify_data.py` checks its complete factor grid and the
reported main-text values; `python figures/plot_from_csv.py` regenerates the
appendix plots. Those checks validate aggregation and plotting, **not** the
historical training runs. One full 200-epoch random-label PLN[4] cell was
compared with its archive; see the [single-cell report](../../docs/CNN_FULL_VALIDATION_20261010.md).
It does not reproduce the three-seed grid or the clean-label setting.
