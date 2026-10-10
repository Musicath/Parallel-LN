# Helmholtz extraction audit

This is a source-level comparison for the standalone
`experiments/pinn/run_helmholtz.py`, not a certificate that it reproduces
every historical Table II result. The historical files below were inspected
read-only in the author-provided `Norm-exp-base` source tree on 2026-10-10.
Only relative source names and hashes are recorded here; private framework
code and raw W&B logs are not redistributed.

| Protocol item | Historical source observation | Standalone candidate |
| --- | --- | --- |
| Table II task | `helmholtz_learnable_2` uses the interval `[-1,1]`, exact solution `sin(pi*x)`, residual `-u''+pi^2*u-2*pi^2*sin(pi*x)`, Dirichlet boundaries, 5,000 domain and 500 boundary points | `make_data(..., "table2", ...)` matches these definitions and counts |
| Separate NTK task | `helmholtz_new` uses the same equation but 100 boundary points | `make_data(..., "ntk", ...)` uses 100; it must not be substituted for Table II |
| Network | Historical MLP has six hidden affine layers, each followed by no-affine PLN or an element-wise activation, then a scalar output | `HelmholtzMLP(depth=6)` retains the corresponding module order and parameter layout; its initial tensors have not been compared for every method |
| SAT | Historical `sinarctan` evaluates `x / sqrt(x^2+1)` | `SinArctan` uses that formula |
| Optimization | Archived launcher sets float64, batch size 128, 6,000 steps, Adam `3e-4` or SGD `1e-4` with momentum `0.9`, and zero weight decay | Table II defaults and optimizer objects match these inspected settings; no learning-rate schedule is applied |
| Scheduler | The historical trainer constructs a step scheduler but the inspected single-stage DeepXDE training path does not advance it | Deliberately omitted; this is the archived-run behavior, not the earlier paper-stated schedule |
| Randomness | Historical `ext.trainer.setting` seeds Python, NumPy and PyTorch before model creation; it enables cuDNN benchmarking | Candidate seeds those generators before model construction; environment-sensitive numerical identity is still unproven for Table II |
| Reporting cadence | Historical Table II launcher uses `display_every=100`; separate NTK launcher uses `10` | Candidate defaults now follow the corresponding protocol; `--display-every` remains overridable |

Source SHA-256 identifiers for the inspected snapshot:

| Relative source path | SHA-256 |
| --- | --- |
| `PDE/pde_dataset.py` | `71f95b5293d804419d29d7014b4afc3c650afb24a73a29a1cc717688f1588c5c` |
| `PDE/pde.py` | `04300b3681db1326e38cba070d186cf78f683d13278a92ea59e2b2455870b0f0` |
| `PDE/reproduce_pln_helmholtz.sh` | `41a1eea1dc280ee0af016be2178c10f5ef244ba8a85976daea0b445fb740e542` |
| `extension/model/mlp/MLP.py` | `37482209d6aa5724daeef7bb19bd62148826e8e281d20d98956eb68feadf629b` |
| `extension/modules/norm/pln.py` | `5656e9a039bceb33ec576a096230bbb42e1a85fb9094ea3c40671205695180e0` |
| `extension/modules/activation/sinarctan.py` | `c56997b1495c8d32e142bf7052a4df8c82d0bf9537b97c300074a7f300591d41` |
| `extension/optimizer.py` | `5b83e1c77c814abd794e22ea3c961b25245d5d593d0c440bcaa80ed2d639fe7c` |
| `extension/trainer.py` | `dcfaf2bf25d6baae19e958d8e3d41ce05fd72ef82fa61ceb7f623c8ab7a2a6b4` |

The 42 selected historical Table II outcomes were audited separately against
their W&B archive. Two 6,000-step PLN[4] cells matched the printed table using
the **original** trainer. The newly extracted Table II runner has also
completed two full 6,000-step selected-cell comparisons:
PLN4/Adam/width 128/seed 1/weights `(1,10)` agrees with the printed
six-decimal cell but differs from the archived raw metric by about 0.13%;
SAT/SGD/width 128/seed 2/weights `(1,10)` agrees with the archived raw metric
to floating-point precision using the corrected initial SGD rate `1e-4`.
The distinct output-NTK extraction has matched one
archived 4,000-step PLN4 row, but that does not close the Table II gate.

All 42 selected W&B metadata records cite a historical Git commit that lacks
the `helmholtz_learnable_2` task. These hashes identify the inspected later
source snapshot; they do not establish exact identity with the uncommitted
source state that produced the original Table II runs.
