# Reproducibility status and commands

This document separates **archived numerical evidence**, **available training code**, and **fresh re-runs**. A plotted CSV or a source file alone is not proof that a historical experiment has been re-run. Status below refers to the expanded arXiv manuscript as inspected during this code release preparation.

| Manuscript item | Training source | Bundled numerical data | Status |
| --- | --- | --- | --- |
| Dense Fourier MLP table, dimensions 4 and 8 | `experiments/dense_fourier/run_dense_fourier_mlp.py` | `figures/data/dense_fourier_d{4,8}_runs.csv` (720 runs per dimension) and matching table summaries | Historical aggregation verified; one full 10,000-step PLN[4] configuration in each dimension closely matched its archived per-run result on RTX 4090 GPU1. The five-target/three-seed paper summary is not freshly reproduced; see `FOURIER_FULL_VALIDATION_20261010.md`. |
| Helmholtz PINN tables | `experiments/pinn/run_helmholtz.py` is a standalone extraction; original `helmholtz_learnable_2` trainer remains outside this release | 42 selected-run configurations and final values in `experiments/pinn/data/table2_selected_runs.csv`; raw W&B logs excluded | Two headline PLN[4] cells matched using the original trainer. Two distinct selected width-128 cells (PLN[4]/Adam and SAT/SGD) completed full 6,000-step runs with the extraction and match printed precision; this is not a full-table reproduction. |
| Exploratory output-NTK appendix | The same extracted runner has a distinct `ntk` protocol and `output_ntk.py` | 240 privacy-screened paired numeric rows for the five manuscript methods in `experiments/pinn/data/ntk_matched_240.csv` | Medians reconstruct the appendix table; the extracted 4,000-step PLN4/width-128/seed-1 run matches one archived row to about `1.61e-16` absolute relative-L2 difference. The rest of the grid remains unvalidated. This is not a PDE-residual or Table II mechanism check. |
| Mixed-axis CNN table and figures | `experiments/cnn/run_cifar10_conv_gn_{random,clean}_labels.py` | `figures/data/cnn_runs.csv` (288 per-seed records) | Data completeness and paper checkpoints pass. [One full random-label PLN[4] GPU cell](CNN_FULL_VALIDATION_20261010.md) matched the archived final fitting accuracy and closely followed its curve; the full grid remains unverified. |
| VGG-16 initialization figure | `experiments/vgg/` contains a minimal extracted model, paper-aligned one-cell runner, non-executing full-factor planner, complete-grid auditor and five-panel plotter; private grid has 630 scripts / two seeds, while manuscript specifies 945 runs / three seeds | Verified historical plotting data not located | CPU checks and limited RTX 4090 training smoke runs passed; newly generated three-seed results would be a replication, not the archived figure, and the original final/best epoch statistic remains unverified |
| ViT table and grouping figures | Extracted `experiments/vit/model.py` and adapted `train_vit_token_grouping.py` | `figures/data/vit_runs.csv` (1,377 per-seed records) plus exact first-6k random labels; original CIFAR-10 clean labels are not bundled | Historical model-equivalence and data checks pass. The minimized label CSV generates byte-identical 6k/20k artifacts to the previous candidate on the same dataset cache. One full 100-epoch GPU1 configuration closely matched its historical best-fit value; see [single-run evidence](VIT_FULL_VALIDATION_20261010.md). Full paper-grid reproduction is pending. |

The coverage table was rechecked against the active experiment sections of the
arXiv source on 2026-10-10. Its active families are Dense Fourier, Helmholtz
PINN, exploratory output-NTK, mixed-axis CNN, VGG-16, and ViT. Earlier MNIST random-label material is
commented out, and the preliminary machine-translation subsection is inside
an excluded `\iffalse` block; neither is a missing release component. This
scope check does not resolve the PINN extraction-validation or VGG figure-provenance gaps above.

## Local checks without training

```sh
python tools/validate_candidate.py
```

This command runs normal and optimized-mode tests and data checks, probes
non-training entry points, and verifies that eight plots were regenerated. It
does not download CIFAR-10, use a GPU, or train a model. The same command is
used by the CPU CI workflow; CI has not run on GitHub for this unpublished
candidate.

`verify_data.py` checks that each expected CNN/ViT dataset--mode--grouping--seed cell occurs exactly once, that source paths are unique, and that reported accuracies are finite percentages. It checks all four printed CNN summary values and all 15 printed ViT summary cells, selecting the best three-seed mean and best individual run independently within each ViT table row. The CNN/ViT plotting entry point applies the same per-run validation before writing any figure. These checks do not audit every original training log. `verify_fourier_data.py` reconstructs the shared-learning-rate selection and two Dense Fourier table summaries from all 1,440 sanitized per-run records, then checks all 32 printed mean/SD pairs against constants independent of the summary CSVs. The plotting scripts write to the ignored `figures/generated/` directory. These figures use compact per-seed CSVs and may differ typographically from manuscript PDF assets. See [PAPER_ARTIFACT_MAP.md](PAPER_ARTIFACT_MAP.md) for item-level coverage.

The experiment/data-preparation entry points passed
`--help` import and argument-parsing checks on Windows. VGG and ViT use
`python -m experiments...`; Dense Fourier uses a direct script path. Both CNN
scripts now support direct paths and `python -m experiments.cnn...`, and both
invocation styles passed argument-parsing checks. These checks neither
download data nor exercise a training epoch.

The [CNN-specific guide](../experiments/cnn/README.md) maps every
`bgn<G>k<K>` name to its normalization domain and gives one complete
random-label and clean-label paper-protocol command. The checked 288-row
historical grid remains separate from any new single-configuration run.

The [ViT-specific guide](../experiments/vit/README.md) covers fixed 6k/20k
artifact generation, the extracted model and one-cell training commands. Its
6k label manifest was checked against an archived run's label counts and first
16 labels. The 20k extension follows the archived preparation algorithm and,
when generated on the eight-RTX-4090 host using an existing CIFAR-10 cache,
matched the archived 20k run's label counts, changed-label count and first 16
clean/random labels. The original 20k artifact digest is not yet available for
bytewise comparison.

The [VGG-specific guide](../experiments/vgg/README.md) records the extracted
model, one-cell runner and prospective grid planner. Its archival grid has only two seeds, and no matching
per-run output logs were found; the current manuscript's 945-run figure is not
validated by the extracted code alone. The historical `val_loader` actually
uses the standard CIFAR-10 test split (`train=False`), which helps interpret
the earlier "validation accuracy" wording but does not recover the figure data.
The archived trainer evaluates that split after every epoch and maintains a
running best test score. Without the plot's original data-selection script,
we cannot determine whether the manuscript figure uses final-epoch or
best-epoch accuracy; the reconstructed runner saves both per epoch.
In a local read-only comparison against an archived `Musicath-base` VGG source,
the extracted PLN model's state keys, initial tensors and forward output were
identical under seed 20 and initialization scale 0.7. This is a model extraction
check, not a training-result comparison. On 2026-10-10, three isolated CIFAR-10
GPU smoke runs from candidate commit `221c94b` completed on an RTX 4090: PLN[8]
and ReLU each trained for one epoch with two training batches; PLN[8] also
completed a ten-epoch warmup with eight training batches per epoch. All used
seed 20. The per-run logs remain outside this repository and have SHA-256
digests `27ccddfa71343fce06ffcb6e0de817f3c382fe503eb3d6a7efed1b6e272a6129`,
`98801e1e01c68cb2dc7706adfbb9a85b494e3bab9bcb4138489d1ce06de0f645`,
and `ff6217b89f55efa4e747917af8114be8e0c38c9285cf97986ca41eaff0154556`,
respectively. Batch-limited accuracy is not comparable to the manuscript figure.

The archived PINN audit traces all 42 Table II values to recorded optimizer/seed
settings. The historical initial learning rates are Adam `3e-4` and SGD `1e-4`
for 6,000 iterations, not `3e-4` for both. The original trainer creates a step
scheduler but does not advance it in the inspected path. Two independent full
PLN[4] re-runs at widths 128 and 256 matched the reported six-decimal table
entries. Additional isolated 100-step Adam and SGD checks completed on an RTX
3090; their console-log SHA-256 digests are
`094811eb97e2d04e644b82baff4dc64b9f8f7b3f26c6dd12bbb30118b958ed67`
and `ef2dbd240d2496bb78ede411199796c90e0ee4d40db192b879a8aa1c458d2610`.
These short runs are execution checks, not table-cell reproductions. The original
one-cell launcher and full-matrix launcher have SHA-256 digests
`41a1eea1dc280ee0af016be2178c10f5ef244ba8a85976daea0b445fb740e542`
and `473fe5833f36440d4a321602358aebfe20c304130f914b1c445d8733593c4252`.
The archived W&B archive's 297 listed file hashes were checked, and all 42
final metrics agree with the printed six-decimal cells. It contains only the
42 selected source runs, not every run in the 420-configuration search. The
raw files carry private person, host, machine-path and run metadata and are
not bundled. `experiments/pinn/data/table2_selected_runs.csv` retains only
the paper-relevant numeric values and launch settings. The historical Git
commit recorded in all 42 runs lacks the `helmholtz_learnable_2` task, so the
exact uncommitted training source state has not been recovered. The standalone
extraction is not claimed to be that source state.

The distinct NTK follow-up is summarized by 240 balanced rows (five methods,
two widths, four Adam rates, three seeds, two optimizer stages) after removing
run identifiers and server paths from a 288-row historical matched matrix.
It uses 100 rather than 500 boundary training points, 4,000 total optimizer
steps, and a 500-point output NTK. The raw NTK results and checkpoints remain
outside this release. `experiments/pinn/verify_data.py` recomputes the
manuscript's per-method medians from the sanitized rows and independently
checks the rounded values printed in both the Table II breakdown and NTK table.

On an idle RTX 3090 GPU1, a separate two-step run of the extracted PLN4
NTK protocol (width 128, seed 1, Adam `1e-4`, `--ntk-points 500`,
`--save-eigvals`) exited successfully in Python 3.10 / PyTorch 2.6.0+cu124 /
DeepXDE 1.15.0. It saved 500 initial and 500 final eigenvalues, with
90% effective ranks 12 and 8. Its finite final relative-L2 metric was
`1.153600329`; the `summary.json` SHA-256 was
`caaadf2c619eeba95a5d6816f2c719185c1e129d6d73e5ca64ec83f51955dad0`.
This is a path/serialization smoke check, not a re-run of the archived
4,000-step matrix. The author-supplied ICML 2026 rebuttal text was checked
against the arXiv discussion; its earlier 64-point PINN and 500-point 8D
MLP ranks lack a complete recoverable protocol and are not promoted to
reproduced results.

The same extracted PLN4/width-128/seed-1 configuration was then run for
the full 4,000 Adam steps, at `1e-4` with 500 output-NTK probe points on
that RTX 3090. Its final relative-L2 error was
`4.452487451562643e-05`, compared with the archived matched row's
`4.452487451578786e-05` (absolute difference about `1.61e-16`). Initial
90%-trace effective rank was 12 in both; the stable-rank diagnostic matched
to printed precision. The full-run `summary.json` SHA-256 was
`f2126a6b997f0abb6875f5c944a1ad7b19e2b345c42a1e27108d8836c1d5a5cb`.
This validates one NTK configuration numerically, not the full grid, all
method families, or the separate 6,000-step Table II protocol.

## Training entry points already included

The scripts below are paper-specific training sources or documented standalone
extractions with machine-independent input/output arguments. The PINN runner
is explicitly a new extraction, not the archived binary. Not every family has
passed a fresh GPU training check, and none of the short checks alone verifies
the full paper grid. Keep results outside the repository or under the ignored
`outputs/` directory.

The [PINN/NTK guide](../experiments/pinn/README.md) gives the two distinct
protocols, one-run commands, original-log provenance caveats, and the
privacy-screened data-release decision.

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

The Dense Fourier runner now refuses to overwrite a result JSON with the same
run key. `--skip-existing` is opt-in and checks every saved experiment
parameter, the evaluation interval, and the resolved device before skipping;
older result files lacking those fields are not silently treated as matches.
The run key alone omits some CLI settings and must not be used as proof that
two runs are identical. Use a different output directory for a deliberate
new run with colliding keys.

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

These CNN scripts use `torchvision.datasets.CIFAR10` but do not contact the
dataset host by default; pass `--download` explicitly if the local cache is
absent. Their per-run JSON records are never overwritten. `--skip-existing`
checks the saved configuration, evaluation interval, device, and resolved data
root before reusing a record; an all-skipped invocation returns before loading
the dataset or requiring a GPU. Older run files without these fields cannot
be silently reused. The training output, downloaded images, and checkpoints
are not part of Git. Review the dataset's own terms before redistributing it.

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
Three VGG tests cover the extracted 13-convolution layout, five activation
families, PLN[8], SAT formula, a forward/backward pass and a synthetic-data
one-epoch file-output check; the total is now fourteen CPU tests. This is not
a fresh CIFAR-10 training check.
Both 6k and 20k fixed-label artifacts were generated in an isolated `/tmp`
directory from the existing CIFAR-10 cache; a second invocation verified them
without overwriting anything.
An archive exported from local commit `ebe3a74` (without the development
directory's ignored files) also passed the eleven CPU unit tests, both data
verifiers and both plotting scripts on Windows. This checks source-tree
portability, but uses an existing Python environment rather than a new
dependency installation.
The same source-tree check was repeated from local commit `0cd15f4` after
the VGG extraction: all fourteen CPU tests, both data verifiers and both
plotting scripts passed. The temporary archive and its generated figures were
removed afterward; this was still not a clean dependency installation.
The current arXiv release candidate was independently exported from commit
`8faaeeb` using `git archive`; from that extracted tree, all sixteen CPU tests,
both data verifiers and both plotting scripts passed on Windows. Import
resolution pointed to the extracted `pln.py`. The temporary archive and
generated figures were removed afterward. This check used an existing Python
environment and is distinct from the earlier isolated Linux dependency
installation at commit `27f170a`.
The extracted VGG model was also copied into the existing temporary validation
tree on the eight-RTX-4090 host. On idle GPU0, a batch-one 32×32 forward pass
returned finite `(1, 10)` logits and the first convolution's backward gradient
was finite. This was **not** a CIFAR-10 training run.
After adding the VGG files and tests to that temporary source copy, the full
fourteen-test suite also passed under the host's existing Python 3.10.16 /
PyTorch 2.4.0 environment. The temporary tree is neither a pristine final
checkout nor a new dependency installation.
`pip check` on this shared conda environment reports an unrelated pre-existing
conflict: `datasets 3.6.0` requires `fsspec[http] <=2025.3.0`, while
`fsspec 2025.3.2` is installed. This repository does not import either
package. The test result is valid as a smoke check of the imported stack, but
the environment must not be represented as a clean, conflict-free install.

On 2026-10-10, a source archive from commit `27f170a` was installed in a
new isolated Linux Python 3.10.16 virtual environment, using PyTorch
2.4.0+cpu, torchvision 0.19.0+cpu and `requirements-smoke.txt`.
`pip check`, all fourteen unit tests, both data verifiers and both plotting
scripts passed. The first installation attempt used only the PyTorch CPU
wheel index and failed because that index did not supply a build dependency;
the successful attempt added PyPI as an extra index. This is an independent
CPU installation and source-tree check, not a full GPU training run or a
reproduction of the manuscript statistics.

A separate fresh Windows Python 3.11.5 environment installed the matching
CPU wheels and passed `pip check`, both data verifiers and both plotting
scripts, but importing PyTorch failed with Windows error 126 while loading
`fbgemm.dll` or one of its dependencies. Consequently, the torch-dependent
unit tests could not start in that environment; this does not establish a
repository-code test failure. The missing Windows runtime dependency has
not been identified, and no third-party DLL was installed.

A separate 51-file snapshot of the **uncommitted candidate at test time** was
assembled from tracked and non-ignored untracked source files outside the
repository on 2026-10-10. With `PYTHONPATH` cleared, it passed all 24 unit
tests, the CNN/ViT, Dense Fourier and PINN/NTK data verifiers, seven training
entrypoint `--help` checks, both PINN protocol dry runs, and both figure
generation scripts. All eight generated figure files were nonempty. This
checks that those imports and non-training workflows do not depend on other
files in the development checkout. It used an existing Windows Python 3.11.5
environment (PyTorch 2.0.1+cpu, torchvision 0.15.2+cpu, NumPy 1.26.4,
Matplotlib 3.7.2), **not** a fresh dependency installation or GPU training
environment. `pip check` in that shared environment reported conflicts among
unrelated installed packages, so it is not a clean-install certificate.

The later 53-file candidate, including the static private-framework import
guard, was transferred as a SHA-256-checked source archive to a separate
temporary Linux directory. The archive digest was
`12967e783dd70082ae84b46b937fe630a54eaf693609962bb60f6ff3a41aa913`.
In an existing Python 3.10.19 / PyTorch 2.6.0+cu124 / torchvision
0.21.0+cu124 / NumPy 2.1.3 / Matplotlib 3.10.7 environment, all 26 tests,
three data verifiers, eight training-entrypoint `--help` checks, both PINN
protocol dry runs, and both plotting scripts passed; eight generated files
were nonempty. This snapshot predates the present documentation note. It
validates source isolation and non-training workflows only: the environment
was not freshly installed from `requirements-smoke.txt`, hosted GitHub CI did
not run, and these checks do not reproduce the manuscript training results.

After that Linux snapshot, all three manuscript-data verifiers were changed
from `assert`-based checks to explicit exceptions: Python's `-O` flag would
otherwise silently remove many completeness and paper-value checks. A source
test now rejects `assert` statements in those verifier files, and a unit test
rejects an altered Table II selected metric. The updated local Windows tree
passed all 28 unit tests in both normal and `python -O` mode, and all three
data verifiers completed in optimized mode. This is a release-integrity fix,
not new evidence that the archived training runs can be reproduced.

The historical Dense Fourier environment (`polynomial`) was also inspected
read-only: Python 3.13.2, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.2.4.
On 2026-10-10, a later exact 64-file candidate snapshot completed isolated
RTX 4090 GPU1 short training in Dense Fourier, CNN, ViT, and VGG paths;
VGG additionally completed ten uncapped epochs with its ten-epoch warmup.
The same snapshot passed the full non-training validator in an existing Linux
environment. Configurations, observed metrics, source/evidence hashes, and
limits are in [GPU_VALIDATION_20261010.md](GPU_VALIDATION_20261010.md).
This is not a fresh dependency installation or a representative full paper
configuration for Dense Fourier, CNN, or ViT. One-seed execution does not
verify three-seed or five-target paper statistics.

After the fixed-label CSV was minimized, an **updated 65-file uncommitted
candidate snapshot** (source archive SHA-256
`756eccbf5a25704ab17f8872f055ba68928c0920d68bebbfba537459daf8a070`)
was installed in a new private Linux Python 3.10.12 venv on the eight-RTX-4090
host. The venv initially had no PyTorch or pip; an existing trusted pip was
directed into it with `pip --python`. Installed versions were PyTorch
2.4.0+cpu, torchvision 0.19.0+cpu, NumPy 2.2.6 and Matplotlib 3.10.3.
`pip check` found no broken requirements, and imports resolved to that venv.
The first validation command attempted `venv/bin/python -m pip check` and
therefore stopped because the no-pip venv did not have a `pip` module; rerunning
the check through the same external pip's `--python` target passed without
reinstalling packages. From the extracted source with `PYTHONPATH` and CUDA
visibility cleared, `tools/validate_candidate.py` passed 44 tests in both
normal and optimized modes, all three data verifiers in both modes, ten
entrypoint checks, two PINN dry runs and eight generated figures. The private
install and validation logs are retained under ignored
`outputs/fresh_candidate_20261010_evidence/`, with SHA-256 values
`0e5b2fe4fac09c20cf514273d4d623e848b812e88e92cb7d6c7f2033ae757878`
and `b33d6e6a7f7986154ade4ae4c5bb738243e49634e6f94e2ffe0e9f61a345152d`.
This fresh install did **not** include DeepXDE, train on a GPU, reproduce
historical metrics, or run GitHub-hosted CI. Its source snapshot predates this
documentation paragraph and is not a final staged release commit.

The same 65-file source snapshot was separately exercised under an existing
Python 3.10.16 / PyTorch 2.6.0+cu126 GPU environment on physical RTX 4090
GPU 1. Full single configurations for Dense Fourier dimensions 4 and 8 and
for the 100-epoch ViT random-20k PLN[4] setting were compared with archived
rows. Their evidence, output hashes and cross-environment caveats are in
[FOURIER_FULL_VALIDATION_20261010.md](FOURIER_FULL_VALIDATION_20261010.md)
and [VIT_FULL_VALIDATION_20261010.md](VIT_FULL_VALIDATION_20261010.md).
Neither check reconstructs the paper's multi-run summary statistics.
The same source snapshot also completed one 200-epoch random-label CNN
PLN[4] cell on GPU 1; its [single-cell report](CNN_FULL_VALIDATION_20261010.md)
records the matched protocol, full-curve comparison and private evidence
hashes. It does not verify other CNN groupings or clean-label runs.
