# Helmholtz PINN and output-NTK experiments

This is a minimal, framework-independent extraction of the paper-specific MLP
and one-dimensional Helmholtz experiment. It uses the repository's `pln.py`,
PyTorch, and DeepXDE; it does not require the private `Norm-exp-base` framework
or a W&B account. The historical runner was more general. This extraction is
**not** the archived training binary, and no short test proves numerical
agreement for every paper cell.

Install a platform-matched PyTorch build first (see `docs/SETUP.md`), then
install `deepxde==1.15.0`. The archived follow-up server environment used
Python 3.10, PyTorch 2.6.0+cu124, and DeepXDE 1.15.0; this is an observed
environment, not a bitwise lockfile for the older Table II runs.

A two-step CPU smoke run of the extracted Table II launcher completed on the
follow-up server (PLN4, width 128, seed 1, Adam `3e-4`, PyTorch 2.6.0+cu124,
DeepXDE 1.15.0). DeepXDE compiled and trained through step 2, and the runner
saved a finite relative-L2 value in `summary.json`. This verifies the basic
training/output path only. Separate full-run comparisons are recorded below.

## Paper Table II protocol

The table uses `helmholtz_learnable_2`, 5,000 interior and 500 boundary
training points, six hidden affine layers of width 128 or 256, float64,
6,000 steps, batch size 128, five seeds `1,2,4,8,42`, three PDE/boundary
loss-weight pairs `(1,10)`, `(1,1)`, `(3,1)`, and seven methods.
The archived initial learning rates are Adam `3e-4` and SGD `1e-4`;
SGD uses momentum `0.9`. A step scheduler was constructed in the old
trainer but was not advanced, so this extraction does not apply a scheduler.

From the repository root, run one cell/seed, for example:

```sh
python experiments/pinn/run_helmholtz.py --protocol table2 --method PLN4 \
  --width 128 --seed 1 --optimizer adam --loss-weights 1 10
python experiments/pinn/run_helmholtz.py --protocol table2 --method Tanh \
  --width 256 --seed 8 --optimizer sgd --loss-weights 3 1
```

The runner defaults to the archived learning rate for the selected optimizer.
Its default evaluation/log interval is 100 steps, matching the historical
Table II launcher; the separate NTK launcher used 10 steps. `--display-every`
can override either value without changing the configured optimization steps.
Use `--dry-run` to inspect the resolved configuration without importing
DeepXDE or training. Outputs are written under ignored `outputs/pinn/`; an
existing run directory is never overwritten. For a complete table search,
run all 7 methods x 2 widths x 3 weight pairs x 2 optimizers x 5 seeds
(420 configurations), then select the lowest final relative L2 error per
method/width/weight pair. The bundled 42 rows in
`data/table2_selected_runs.csv` identify **only the selected source runs**,
not all 420 search logs. They contain no run URLs, people, machine paths,
hardware identifiers, or authentication material.

On 2026-10-10, two selected width-128 cells were independently run for all
6,000 steps on RTX 3090 GPU1 with the standalone extraction (Python 3.10.19,
PyTorch 2.6.0+cu124, DeepXDE 1.15.0). Both used seed-controlled float64,
500 boundary points, loss weights `(1,10)`, and no scheduler:

| Method / optimizer / seed | Initial LR | Extracted relative L2 | Archived selected-run value | Printed cell |
| --- | ---: | ---: | ---: | ---: |
| PLN4 / Adam / 1 | `3e-4` | `3.512519777042503e-05` | `3.5079111943428005e-05` | `0.000035` |
| SAT / SGD / 2 | `1e-4` | `0.00010381260104261615` | `0.00010381260104261616` | `0.000104` |

The PLN4 raw values differ by `4.6085826997025e-08` (about 0.13% of the
archived value); they agree only at the paper's six-decimal precision. The SAT
values differ at floating-point rounding scale. The exact five-file source
archive used for both runs has SHA-256
`d25f07fd9c878a040b26bc55beaaca1b56645a8e95b18120c86593e4e4fe153a`.
The generated PLN4 and SAT `summary.json` digests were, respectively,
`5463c1b2471acbebc7e9b7760ee318fd3b7fff31591ce5a908e31f9fbc7cc0ca`
and `d04cc4be5a349baa8b5780e5d54e5a85438bd7370890726df28d0ec5ae56e67c`.
The two newly generated logs and summaries are preserved in the local ignored
`outputs/pinn_table2_validation_20261010/` directory for author review; they
are not part of the release candidate. The remote temporary run directory was
removed after hash-verified transfer.
These comparisons validate two selected cells, not the other 40 selected
values, the 420-configuration search, or bitwise identity with the historical
trainer.

## Separate output-NTK follow-up

The exploratory NTK results in the arXiv appendix use `helmholtz_new`: 5,000
interior and **100** boundary training points, `(1,10)` loss weights, widths
128/256, Adam rates `1e-5,3e-5,1e-4,3e-4`, seeds `1,2,4`, and either 4,000
Adam steps or 2,800 Adam steps plus at most 1,200 L-BFGS steps. The old
L-BFGS configuration used `maxiter=20`. The output NTK is `K = J J^T`, where
each row of `J` differentiates the scalar network output against parameters
at one of 500 uniformly spaced interior points. Its 90% effective rank is
the number of leading eigenvalues needed to reach 90% of the trace. This is
**not** the Jacobian of the PDE residual or weighted boundary loss.

Example one-run commands:

```sh
python experiments/pinn/run_helmholtz.py --protocol ntk --method PLN4 \
  --width 128 --seed 1 --lr 1e-4 --steps 4000 --save-eigvals
python experiments/pinn/run_helmholtz.py --protocol ntk --method SAT \
  --width 128 --seed 1 --lr 1e-4 --steps 2800 --lbfgs-steps 1200
```

The manuscript's five-method summary uses the 240 numeric rows in
`data/ntk_matched_240.csv`, a privacy-screened subset of the historical
288-run matrix. These are **archived outcomes**, not output of this new
extraction. The omitted sixth family (PQN) is not used in the manuscript
table. `verify_data.py` checks the balanced grid, recomputes its medians,
and checks the rounded manuscript cells independently of the CSV. It also
rejects non-finite metrics and checks that the
paired Adam-only and Adam--L-BFGS rows retain identical initialization
diagnostics for each width, learning rate, and seed:

```sh
python experiments/pinn/verify_data.py
```

Output-kernel effective rank is not a monotone predictor of PINN error in
these records; the NTK experiment must not be cited as a proof of the paper's
universal-approximation theorems or as an explanation of the different
Table II protocol.

On 2026-10-10, the extracted NTK path completed a separate two-step PLN4
smoke run on an idle RTX 3090 GPU with Python 3.10, PyTorch 2.6.0+cu124,
and DeepXDE 1.15.0. It used width 128, seed 1, Adam `1e-4`, 100 boundary
training points, and the full 500-point output kernel. Both initial and final
500-eigenvalue arrays were saved; the 90% effective ranks were 12 and 8,
respectively. The final relative-L2 metric was finite (`1.153600329`), as
expected for a two-step smoke run, not a converged experiment. The generated
`summary.json` SHA-256 was
`caaadf2c619eeba95a5d6816f2c719185c1e129d6d73e5ca64ec83f51955dad0`.
This verifies execution and output structure only, **not** agreement with
the archived 4,000-step matrix or the manuscript's summary medians.

The same extracted PLN4 configuration was then trained for the full 4,000
Adam steps on that GPU (width 128, seed 1, learning rate `1e-4`, 500 NTK
probe points). Its final relative-L2 error was
`4.452487451562643e-05`, versus `4.452487451578786e-05` in the archived
matching row: an absolute difference of about `1.61e-16`. The initial
90%-trace effective rank was 12 in both runs, and the stored stability-rank
diagnostic also agreed to printed precision. The full-run `summary.json`
SHA-256 was
`f2126a6b997f0abb6875f5c944a1ad7b19e2b345c42a1e27108d8836c1d5a5cb`.
This is strong numerical evidence for **one** extracted NTK configuration;
it does not validate the other methods, optimizer stage, full 240-row grid,
or the distinct Table II training protocol.

The ICML 2026 rebuttal reported exploratory effective ranks for a
64-point one-dimensional PINN probe and a 500-point eight-dimensional
random-input MLP. Those rank values are not included in the arXiv table or
this data release because the original probe sampling, initialization,
seeds, and eigenspectra have not been recovered. The rebuttal's inference
that a larger effective rank implies a better-conditioned kernel or
guarantees easier training is not supported by this metric alone.

## Provenance and publication decision

The historical Table II W&B archive contains the 42 selected commands,
configs, sampled metric histories, and console logs. Its SHA-256 is
`1e5bbc82f6acfc88b86be0732f5e7d6d6506e5ba5fa454e162ab31ea2ba294ad`.
All 297 archive-file hashes were checked, and each selected final metric
rounds to the printed manuscript value. The **raw archive is intentionally
excluded** because its metadata contains names/addresses, host information,
machine paths, and run URLs; the two sanitized CSVs above provide the
numerical evidence relevant to the paper without those identifiers.

All 42 run records name one historical Git commit, but the tree at that
commit does not contain the `helmholtz_learnable_2` task. The full uncommitted
historical source state therefore has not been recovered. The current
project's task definition and archived commands were used to make this
standalone extraction. Do not claim exact source-level identity or full-grid
reproduction until a clean re-run confirms it.
