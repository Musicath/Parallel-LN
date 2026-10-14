# Historical 108-cell VGG-Bias initialization study

This is a small, framework-independent extraction of the **separate**
`Musicath-base` VGG-Bias initialization sweep at commit
`6647574352bbbb3e9b76449097647c0d21b91fe0` (2026-05-29). It is **not**
the arXiv manuscript's PLN[8]/ReLU/SiLU/Tanh/SAT VGG figure, which specifies
945 runs. No original 108-cell run logs or Figure data are supplied here, and
the extracted code has not completed the 108-cell training grid.

The grid is ReLU/Tanh × No/LN/BN × initialization scalar
`{0.5, 1, 2}` × bias-variance fraction `{0, .2, .4, .6, .8, 1}`.
The historical launcher uses one fixed seed, `42`; this is **108 configurations,
not 108 × 3 seed runs**. The model has 13 bias-enabled convolutional layers,
four max pools, a 512-wide bias-enabled fully connected block, and a final
linear classifier. `LN` in this code is the historical `GroupNorm(1, C)`
operator over channels **and spatial positions**, not position-wise PLN.

The normal initializer sets weight variance to
`(1-beta)*scalar/fan_in` and bias variance to `beta*scalar`. The 2026-05
optimizer state used SGD, LR `0.03`, momentum `0.9`, and weight decay `1e-4`
on **all** trainable parameters. The current `Musicath-base` optimizer was
changed in July to exclude normalization and bias parameters from decay;
this extraction follows the earlier source. It uses the archived CIFAR-10
augmentation, 100 epochs, and no label smoothing.

The historical YAML and its committed parser disagree. The YAML places batch
size `512`, workers `4`, step milestones `20/40/60/80`, and a 10-epoch warmup
in fields that the committed parser did **not** map. The extracted runner
therefore provides two explicit profiles:

| Profile | Batch / workers | Learning rate |
| --- | --- | --- |
| `committed-parser` (default) | `64` / `2` | cosine from `0.03`, no warmup |
| `yaml-intended` | `512` / `4` | 10-step warmup, then milestones `20/40/60/80` relative to its end |

The archived warmup wrapper in `yaml-intended` starts the first epoch at the
base LR and then sets epoch 1 to `0.0057`; this unusual behavior is preserved.
Neither profile is certified as the exact protocol of the historical report,
because the committed launcher itself has broken paths.
Use a different `--output-root` when planning both profiles; their cell names
otherwise intentionally match the historical factor names.

Generate a plan without training:

```sh
python -m experiments.vgg_bias_init.make_plan \
  --data-root data/cifar10 --plan outputs/vgg_bias_init/plan.jsonl
```

Run one selected cell after reviewing compute and data availability:

```sh
python -m experiments.vgg_bias_init.train \
  --data-root data/cifar10 --output outputs/vgg_bias_init/No_relu_b0.0_s1.0 \
  --norm No --activation relu --scalar 1.0 --beta 0.0 \
  --profile committed-parser --seed 42 --epochs 100 --device cuda:0
```

The output directory must not already exist. Each run writes a configuration
and per-epoch train/test metrics, including final and running-best test
accuracy. The old repository called the standard CIFAR-10 test split
`val_loader`; it is not an independent validation split. `--max-train-batches`
is solely a smoke-test option and makes accuracy non-comparable.

Provenance limit: the committed historical launcher points at nonexistent
relative paths for `runner.py` and its YAML configuration. This extraction
resolves those references by implementing the protocol recorded in the
launcher, configuration, model, task, loader, initializer, scheduler and
optimizer at the identified commit, **while exposing both conflicting
configuration interpretations**. It cannot establish which exact working
tree produced the old report. The old numerical results must not be described
as reproduced until a full rerun is compared against authenticated records.
