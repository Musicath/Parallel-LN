# Parallel Layer Normalization

PyTorch implementation and research code for *Parallel Layer Normalization for Universal Approximation*. This repository retains the original `ParallelLN` API and is being expanded to accompany the arXiv version of the paper. The manuscript and its LaTeX sources are maintained separately.

**Scope:** this arXiv companion includes code for the active experiments and sanitized numerical records. The original PINN records have been audited, and a standalone PINN/output-NTK extraction is included. One extracted NTK configuration and two Table II cells have passed full-run comparisons; the remaining grids have not. The VGG figure's complete historical run records remain missing, so its standalone code is a runnable reconstruction, not a reproduction of that figure. Checks and limits are documented in the [reproducibility status](docs/REPRODUCIBILITY.md); this release does not reproduce every manuscript result.

## The operator

`ParallelLN` splits the feature coordinate into consecutive blocks of fixed size and applies Layer Normalization independently to each block. For an image tensor, it does this at each spatial position. It is **not** conventional Group Normalization, which also pools over spatial positions. `centering=False` gives the corresponding RMS-style Parallel Layer Scaling (PLS). `eps` and learnable affine parameters are practical implementation details; the paper's idealized mathematical operator omits them.

```python
import torch
from pln import ParallelLN

x = torch.randn(2, 16)
y = ParallelLN(16, num_per_group=4, dim=2)(x)

images = torch.randn(2, 32, 16, 16)
features = ParallelLN(32, num_per_group=8, dim=4)(images)
```

Supported layouts are `[batch, features]` (`dim=2`), `[batch, tokens, features]` (`dim=3`), and `[batch, features, height, width]` (`dim=4`). The feature count must be divisible by `num_per_group`. `example.py` retains the original MLP/CNN usage examples.

## Repository map

| Area | Contents | Current status |
| --- | --- | --- |
| `pln.py`, `tests/` | Position-wise PLN/PLS and numerical tests | CPU tests pass |
| `experiments/dense_fourier/` | Dense Fourier MLP training source | Matching warmup-cosine source and 1,440 per-run records included; two full single-config GPU runs closely matched archived rows, not the full table |
| `experiments/cnn/` | Clean- and random-label CIFAR-10 mixed-axis CNN runners | Source and [paper-protocol guide](experiments/cnn/README.md) included; [one full random-label GPU cell](docs/CNN_FULL_VALIDATION_20261010.md) matched archived final fitting accuracy, but the full grid remains unverified |
| `experiments/vit/` | Minimal ViT backbone, fixed labels and grouping runner | Extracted model verified against historical initialization/output; [one full GPU configuration](docs/VIT_FULL_VALIDATION_20261010.md) closely matched its archived best-fit accuracy; the full grid remains unverified |
| `figures/` | Portable CNN/ViT plotting scripts and compact per-seed CSVs | Data checks pass; plots can be regenerated |
| `experiments/pinn/` | Standalone Helmholtz PINN and output-NTK candidate, with 42 selected Table II records and 240 sanitized NTK rows | One full NTK configuration and two selected full Table II cells matched archived or printed values; the remaining grids are not validated |
| `experiments/vgg/` | Minimal VGG-16 extraction, reconstructed one-cell runner and non-executing grid planner | CPU checks and limited GPU training smoke runs pass; historical 945-run figure/data still unverified |

The translation experiments are not part of the active arXiv manuscript and are not included here.

## Quick checks

Use a compatible Python/PyTorch/torchvision environment; a tested setup and
platform-specific notes are in [docs/SETUP.md](docs/SETUP.md). Then run:

```sh
python tools/validate_candidate.py
```

This non-training command runs tests, numerical-data checks, entrypoint checks
and plot generation. It is not a fresh-install check or a reproduction of the
training experiments. The generated figures are written to `figures/generated/` and are intentionally not tracked. The [paper artifact map](docs/PAPER_ARTIFACT_MAP.md) gives a table/figure-level correspondence and identifies the missing VGG figure data. Training environments and limitations are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); outstanding publication gates are in the [release checklist](docs/RELEASE_CHECKLIST.md). CIFAR-10 images and model checkpoints are not redistributed.

The CPU checks are also configured in a GitHub Actions workflow. Hosted results should be checked on the repository's Actions page; the local reports do not certify a hosted run.

The [ViT guide](experiments/vit/README.md) explains how to reconstruct the fixed-label artifacts without publishing CIFAR-10 images or the private `Musicath-base` framework.

## References and license

1. Ni et al., *On the Nonlinearity of Layer Normalization*, arXiv:2406.01255 (2024).
2. Ni et al., *Parallel Layer Normalization for Universal Approximation*, [arXiv:2505.13142](https://arxiv.org/abs/2505.13142) (2025; the arXiv revision is in preparation).

If you use this repository for work reported in the paper, please cite the
arXiv paper above. A GitHub `CITATION.cff` is deferred: the CFF format requires
software-author metadata at the root, and the author has not yet designated
software authors for this expanded repository. The paper's author list is not
being treated as a software-author list.

The original repository uses the [Apache License 2.0](LICENSE.txt). The author has also approved release of the nine candidate CSVs under Apache-2.0; the [data-release record](docs/DATA_RELEASE_REVIEW.md) identifies their contents and the separate CIFAR-10 attribution. This does not license or redistribute CIFAR-10 itself. Source provenance and third-party licensing checks for the expanded research code are tracked in [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md), with ViT upstream attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Inclusion here should not be read as a claim that every paper result has been re-run.
