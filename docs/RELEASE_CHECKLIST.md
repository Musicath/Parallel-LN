# arXiv code-release checklist

This is a release audit, not a claim that every manuscript result
has been independently reproduced. The public repository is
`Musicath/Parallel-LN`; the ICLR 2027 submission remains separate and should
not link to this repository while under its own review policy.

**Author release decision (2026-10-10):** the author lifted the earlier
publication hold and authorized a public push after final file and source
checks. The original PINN records and scripts have been located; a standalone
PINN/output-NTK extraction and sanitized data are now present. One full NTK
configuration and two selected Table II cells have been checked with the
standalone extraction; the remaining grids still need validation. The
complete historical VGG run records remain missing, but the author later
clarified that releasing the VGG plotting data is not required if the
experiment code is included. This does **not** turn the historical figure into
a reproduced result. The final release audit must keep these limitations visible.

| Area | Current evidence | Residual limitation or action |
| --- | --- | --- |
| Core PLN/PLS operator | API retained; numerical, gradient and shape tests pass | Confirm final arXiv notation and examples |
| Dense Fourier | Matching warmup-cosine runner; 720 sanitized runs per dimension reconstruct both paper summaries. Full 10,000-step PLN[4] configurations for dimensions 4 and 8 from the same isolated candidate closely matched their archived per-run values on RTX 4090 GPU1; environment, numerical differences and hashes are in `FOURIER_FULL_VALIDATION_20261010.md`. | The remaining target/seed/learning-rate grid has not been freshly run. Do not describe these two cells as reproduction of either complete manuscript table. |
| CNN mixed-axis grid | Source, 288 per-seed data rows, figure generation and GN/PLN endpoint tests. Both runners require explicit `--download`, protect per-run JSON from overwrites, and check all saved parameters before `--skip-existing`. A full 200-epoch random-50k PLN[4] GPU1 cell from the isolated candidate matched the archived 100% final fitting accuracy and had a closely aligned curve; the [report](CNN_FULL_VALIDATION_20261010.md) records config, environment and hashes. | The other random-label grouping/seed cells and all clean-label cells have not been freshly verified; the saturated single-cell result does not establish the main table or appendix grid. |
| ViT grouping | Minimal extracted model matches historical parameters/output; 1,377 per-seed rows pass checks. Fixed 6k/20k artifacts regenerate; after removing original CIFAR-10 clean labels from the tracked CSV, both sizes' tensors, summaries and metadata matched the previous candidate byte-for-byte on the same dataset cache. A full 100-epoch random-20k position-wise PLN[4] configuration on GPU1 reported 81.940% best fitting accuracy versus 82.425% historically; source/environment, output hashes and caveats are in [VIT_FULL_VALIDATION_20261010.md](VIT_FULL_VALIDATION_20261010.md). | The original historical 20k artifact digest remains unavailable; other configurations and the three-seed paper results are not verified by this one full run. |
| VGG-16 stability figure | Minimal model and reconstructed runner added; seed-20 initialization/output match an archived private VGG source. A non-executing planner enumerates the factor grid. Beyond earlier batch-limited smoke runs, the isolated candidate completed one uncapped epoch and ten uncapped epochs with the ten-epoch warmup on GPU1; private logs/configs are preserved and hashed in `GPU_VALIDATION_20261010.md`. The private grid still has only 630 scripts for two seeds, without per-run logs; the historical third seed and figure data remain unidentified. The author waived shipping VGG plotting data if experiment code is included. | Release the runner as a runnable reconstruction, explicitly **not** a reproduction of the historical 945-run figure. Do not infer whether that figure used final-epoch or best-epoch accuracy. Missing historical plot data are a provenance limitation, not by themselves a code-publication blocker after the author's waiver. |
| Helmholtz PINN | All 42 selected table runs and 297 archive-file hashes checked. Adam `3e-4`, SGD `1e-4`; original-trainer PLN[4] headline reruns matched printed precision. A standalone candidate and privacy-screened 42-row CSV are included; the verifier checks every printed Table II cell. Two full 6,000-step extracted runs on an RTX 3090 now cover PLN[4]/Adam and SAT/SGD at width 128, weights `(1,10)`; both match the printed cells, with the PLN[4] raw error differing by about 0.13% and the SAT raw error matching to numerical precision. See `REPRODUCIBILITY.md` and `PINN_EXTRACTION_AUDIT.md`. | The other 40 selected cells and full 420-configuration search are not reproduced. Preserve the historical source-state caveat and check release rights; raw W&B logs contain private metadata and are excluded. Do not substitute the later k=1/k=4 protocol. |
| Exploratory output NTK | The new arXiv appendix uses a distinct protocol; 240 balanced, sanitized numeric rows for its five methods reconstruct the reported medians. The verifier checks finite values, paired initialization metrics and each rounded printed table cell. A full 4,000-step extracted PLN4/width-128/seed-1 run on an RTX 3090 matched the archived relative-L2 error to about `1.61e-16` absolute difference and matched initialization rank diagnostics. The ICML 2026 rebuttal text supplies earlier exploratory 64-point PINN and 500-point 8D-MLP ranks but no complete configurations or original eigenspectra. | Check additional methods/optimizer stages before claiming grid reproduction. Do not import the unverified rebuttal ranks or infer conditioning/training superiority from effective rank. Retain the caveat that the output NTK is not a PDE-residual mechanism or a monotone error predictor. |
| Legacy MNIST and translation | MNIST random-label material is commented out; preliminary translation is in an excluded `\iffalse` block of the arXiv source | Keep excluded unless the author activates either experiment in a later revision |
| Source/licensing | Original repository has Apache-2.0; the author authorized a minimal paper-specific extraction while retaining the broad `Musicath-base` framework. Added files are not all newly written. The historical ViT backbone overlaps timm; its official source names Ross Wightman (copyright 2020) and its repository carries Apache-2.0. The extracted ViT file has a file-level notice. The private VGG source resembles torchvision's VGG-D configuration and initializer; `THIRD_PARTY_NOTICES.md` records conservative attribution, and the upstream BSD-3-Clause text is retained. The official current upstream license pages were rechecked before publication. A source-AST test rejects static imports of known private frameworks; inspection found no private absolute path defaults. | Exact historical upstream revisions and ownership of every incorporated line cannot be certified from the available archive. Preserve the notices and seek stronger lineage evidence if it becomes available; this audit is not a legal determination. |
| Data-release rights | The [data-release review](DATA_RELEASE_REVIEW.md) inventories all nine candidate CSVs. Eight contain configurations/numeric experimental records; the ninth now contains only 6,000 CIFAR-10 indices and the experiment's fixed random labels. Original clean labels are derived at runtime from the user's CIFAR-10 copy and checked by digest. No CIFAR-10 images, original per-example labels, or raw private run logs are bundled. On 2026-10-10 the author expressly approved public release of all nine CSVs under Apache-2.0. The exact staged data passed the manifest and semantic checks. | Retain the separate CIFAR-10 attribution; this repository does not redistribute or license the underlying CIFAR-10 dataset. |
| Citation metadata | The author chose arXiv:2505.13142 as the preferred citation and asked not to fill software-author metadata yet. The README now directs readers to the paper. CFF 1.2.0 requires root-level software authors even when `preferred-citation` points to a paper, so no `CITATION.cff` has been created or falsely attributed to the paper authors. All relative Markdown documentation links resolve locally. | If a GitHub citation prompt is desired, the author must later designate software author(s) or an appropriate project entity; then add and validate `CITATION.cff` with the paper as `preferred-citation`. README citation guidance can stand without it. |
| Sensitive material | The staged 70-file tree and all 23 outgoing commit trees were screened on 2026-10-10 for common token/private-key markers, private absolute paths and the known server host; no matching path was found. The exact staged diff against `origin/main` changes 68 files, has no whitespace errors, symlinks or submodules, and contains no private outputs, datasets or checkpoints. The 23 commits' changed-path inventory and author metadata were inspected; 91 outgoing blobs were counted, with maximum size 263,455 bytes and none above 10 MiB. No candidate file exceeds 10 MiB. The ViT runner omits changed filenames from saved Git metadata. | Pattern and size checks are **not** a complete secret or rights audit. Ignore rules cannot protect material committed in an earlier history, and user-supplied paths may still appear in future run configurations. |
| Shipped numerical-data integrity | Nine compact CSVs have SHA-256 entries in `docs/DATA_MANIFEST.sha256`; a unit test checks complete CSV coverage, LF line endings and exact bytes. Five Windows CRLF working copies were normalized to the repository's declared LF format, and the manifest now hashes the checkout-stable bytes without changing data rows. CNN/ViT semantic checks require every expected per-seed factor cell exactly once, unique source paths, and finite accuracy percentages; corruption tests reject a duplicate replacing a missing run and an out-of-grid ViT cell. | Treat the digest list as release-file identity, not proof of the historical raw experiments; refresh it only after auditing any deliberate data change |
| Validation | The post-label-minimization 65-file uncommitted snapshot passed a **fresh isolated Linux CPU installation** (Python 3.10.12, PyTorch 2.4.0+cpu, torchvision 0.19.0+cpu, NumPy 2.2.6, Matplotlib 3.10.3), `pip check`, 44 tests in normal and optimized modes, all three data verifiers in both modes, ten entrypoint checks, two PINN dry runs and eight figures; source/log hashes and the no-pip-venv bootstrap caveat are in `REPRODUCIBILITY.md`. The current 70-file staged candidate matched the working tree and passed 45 tests in both modes, including a local-documentation-link check, plus the same verifiers, entrypoint checks and eight regenerated plots on the development machine. An earlier 64-file snapshot passed short Dense Fourier, CNN, ViT and uncapped VGG runs on RTX 4090 GPU1 (`GPU_VALIDATION_20261010.md`). The 65-file snapshot completed full single-cell Dense Fourier d4/d8, ViT and CNN checks on GPU1 (separate reports). Old/new 6k and 20k fixed-label artifacts matched byte-for-byte. Two extracted full Table II PINN cells and one full NTK cell were separately compared on RTX 3090. | The fresh CPU venv did not include DeepXDE. A fresh Windows Python 3.11 installation could not import PyTorch because of unresolved `fbgemm.dll`; do not claim Windows validation. These checks do not reproduce full paper statistics. |
| Continuous integration | CPU workflow runs normal/optimized-mode tests, data checks and plots. The first [GitHub-hosted run](https://github.com/Musicath/Parallel-LN/actions/runs/38058919256) on release commit `6a72b03` completed successfully, including `cpu-checks`. | This hosted CPU run does not reproduce GPU training or the complete paper grids; rerun CI for later commits. |
| GitHub transport | The author authorized publication after final review. Commit `6a72b036de95169cc6044dec9378faa464197840` was pushed to `Musicath/Parallel-LN` `main` on 2026-10-10, and `git ls-remote` returned the same head. | Keep the repository and subsequent arXiv manuscript link in sync; do not link the separate ICLR submission while its review policy applies. |
| Paper linkage | arXiv manuscript is maintained separately | Add repository link and availability statement only after the public release is actually available |

A read-only bytewise comparison against the exact 65-file archive used for the
full GPU checks found no changes to any archived source code, data, tests,
configuration or license file. Seven archived Markdown files changed; the
current 70-file candidate additionally contains four Markdown files (three
GPU reports and the CNN experiment guide) and one documentation-link test.
Thus the full GPU runs still apply to the unchanged experiment code, while the
new test, documentation and eventual staged package require final review.

The active experimental table/figure coverage and its evidence limits are
itemized in [PAPER_ARTIFACT_MAP.md](PAPER_ARTIFACT_MAP.md). In particular, the
candidate can regenerate seven manuscript-named schematic/CNN/ViT PDFs with
matching extractable text, but the regenerated files are not byte-identical to
the manuscript assets, and the VGG figure cannot yet be reconstructed from
verified per-run data.

On 2026-10-10, the 64 non-ignored candidate files reported by `git ls-files
--cached --others --exclude-standard` were copied into a separate directory
without `.git` or ignored files. Every copied file matched its source SHA-256.
With `PYTHONPATH` cleared, `python tools/validate_candidate.py` passed there
in the existing Windows Python 3.11.5 / PyTorch 2.0.1+cpu environment. This
checks source-tree independence for the non-training validator only. It is not
a clean dependency installation, a GPU training check, or proof that the exact
candidate will run under the proposed Python 3.10 / PyTorch 2.4 stack.

The same 64-file candidate was then packaged with SHA-256
`921a2923a17b384f33ccd13bae57434c8f710fb84554dcb03d3e37ae556b842b`
and checked in a private temporary directory on the eight-RTX-4090 Linux
node. The archive hash and extracted file count matched. With
`CUDA_VISIBLE_DEVICES` and `PYTHONPATH` cleared, the non-training validator
passed two 43-test runs, both modes of each numerical-data verifier, all
entrypoint probes and eight regenerated plots under an **existing** Python
3.10.16 / PyTorch 2.6.0+cu126 / torchvision 0.21.0+cu126 environment. This
is an exact-candidate source check, not a fresh dependency installation or a
GPU experiment. At inspection time, GPUs 0 and 1 were occupied, so no new
training was started.

A later 64-file candidate snapshot, SHA-256
`76edac57e3fab447269b2ae5b9ce4d8969d20d55971c80c671a023e142015eee`,
was installed in a **new** Python 3.10.12 virtual environment in the node's
shared memory, with the official PyTorch 2.4.0+cpu / torchvision 0.19.0+cpu
pair and the repository's pinned NumPy 2.2.6 / Matplotlib 3.10.3. Its
dependency check reported no broken requirements. With GPU visibility and
`PYTHONPATH` cleared, the same candidate validator passed two 43-test runs,
both modes of all three data verifiers, the entrypoint checks, and eight
regenerated plots. The server's system Python lacked `ensurepip`, so an
existing pip 25.0 used `--python` to install into the otherwise new venv;
packages were not inherited from its source environment. This closes the
fresh **CPU** installation check for that snapshot, not a GPU training or
DeepXDE installation check. Recheck the exact staged release files after
later edits; do not transfer this result to a changed candidate without a
new run.

No claim here implies that single-run smoke checks reproduce three-seed
statistics. Use `REPRODUCIBILITY.md` for exact protocols and caveats and
`SOURCE_PROVENANCE.md` for original source identifiers.
