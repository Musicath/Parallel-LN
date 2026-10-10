# arXiv code-release checklist

This is a working release checklist, not a claim that every manuscript result
has been independently reproduced. The planned public repository is
`Musicath/Parallel-LN`; the ICLR 2027 submission remains separate and should
not link to this repository while under its own review policy.

| Area | Current evidence | Remaining gate |
| --- | --- | --- |
| Core PLN/PLS operator | API retained; numerical, gradient and shape tests pass | Confirm final arXiv notation and examples |
| Dense Fourier | Matching warmup-cosine runner; 720 sanitized runs per dimension reconstruct both paper summaries | Fresh representative GPU run and exact environment note |
| CNN mixed-axis grid | Source, 288 per-seed data rows, figure generation and GN/PLN endpoint tests | Fresh representative GPU run |
| ViT grouping | Minimal extracted model matches historical parameters/output; 1,377 per-seed rows pass checks; fixed 6k/20k artifacts regenerate and match archived label summaries | Fresh representative GPU run; original 20k artifact digest if available |
| VGG-16 stability figure | Minimal model and reconstructed runner added; CPU structure/gradient tests pass and seed-20 initialization/output match an archived private VGG source. Private `soft-norm` grid has 630 scripts for seeds 20 and 30 but no matching per-run logs were found. An earlier manuscript describes validation accuracy, although the relevant historical loader uses CIFAR-10 `train=False` | Locate the third seed and per-run results, verify figure metric and seed IDs; run a short GPU training check; neither the 108-run nor incomplete 630-script source is evidence for the 945-run figure |
| Helmholtz PINN | Historical manuscript table is documented, but exact logs/runner are missing | **Paused at author's request** until original logs or scripts are provided; no alternate-protocol data may be substituted |
| Translation | Outside the active manuscript | Keep excluded |
| Source/licensing | Original repository has Apache-2.0; the author requested extracting only the minimal paper path while retaining the broad `Musicath-base` framework | Final authorship/permission review of every added script before public push |
| Sensitive material | Current added text/code scan found no private absolute paths or embedded credentials; large checkpoints and CIFAR images are excluded | Re-scan the exact staged diff immediately before push |
| Validation | Earlier clean source archive passed 11 tests, data verifiers and plotting; current working tree passes 14 tests including VGG. Remote GPU0 operator/ViT gradient and fixed-label artifact checks passed | Repeat clean-archive check on final tree, new dependency installation check and at least one short train-run smoke test per publishable family |
| Continuous integration | CPU workflow added for tests, data checks and plots | Its first GitHub-hosted run must pass after public push; no hosted run is claimed yet |
| Paper linkage | arXiv manuscript is maintained separately | Add repository link and availability statement only after the public release is actually available |

No claim here implies that single-run smoke checks reproduce three-seed
statistics. Use `REPRODUCIBILITY.md` for exact protocols and caveats and
`SOURCE_PROVENANCE.md` for original source identifiers.
