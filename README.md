# Parallel Layer Normalization

PyTorch implementation and research code for *Parallel Layer Normalization for Universal Approximation*. This repository retains the original `ParallelLN` API and is being expanded to accompany the arXiv version of the paper. The manuscript and its LaTeX sources are maintained separately.

**Release status:** preparation in progress. The core operator, extracted ViT model, and bundled figure data have CPU checks; the training runners have not yet passed a fresh-machine GPU reproduction. See [reproducibility status](docs/REPRODUCIBILITY.md) before using the numerical results.

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
| `experiments/dense_fourier/` | Dense Fourier MLP training source | Matching warmup-cosine source and 1,440 per-run records included; GPU re-run pending |
| `experiments/cnn/` | Clean- and random-label CIFAR-10 mixed-axis CNN runners | Source included; GPU reproduction pending |
| `experiments/vit/` | Minimal ViT backbone, fixed labels and grouping runner | Extracted model verified against historical initialization/output; GPU reproduction pending |
| `figures/` | Portable CNN/ViT plotting scripts and compact per-seed CSVs | Data checks pass; plots can be regenerated |
| Helmholtz PINN | Paused pending the authors' original logs or scripts | Not yet claimed reproducible |
| VGG-16 training | Architecture source identified; exact 945-run sweep/data still under review | Not yet release-ready |

The translation experiments are not part of the active arXiv manuscript and are not included here.

## Quick checks

Use a compatible Python/PyTorch/torchvision environment; a tested setup and
platform-specific notes are in [docs/SETUP.md](docs/SETUP.md). Then run:

```sh
python -m unittest discover -s tests -v
python figures/verify_data.py
python figures/verify_fourier_data.py
python figures/plot_from_csv.py
python figures/plot_normalization_domain_schematic.py
```

The generated figures are written to `figures/generated/` and are intentionally not tracked. Details of the training environments, experiment-to-paper map, and limitations are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); outstanding publication gates are in the [release checklist](docs/RELEASE_CHECKLIST.md). CIFAR-10 images and model checkpoints are not redistributed.

The CPU checks are also configured in a GitHub Actions workflow; its first hosted run is pending publication.

The [ViT guide](experiments/vit/README.md) explains how to reconstruct the fixed-label artifacts without publishing CIFAR-10 images or the private `Musicath-base` framework.

## References and license

1. Ni et al., *On the Nonlinearity of Layer Normalization*, arXiv:2406.01255 (2024).
2. Ni et al., *Parallel Layer Normalization for Universal Approximation*, arXiv:2505.13142 (2025; the arXiv revision is in preparation).

The original repository uses the [Apache License 2.0](LICENSE.txt). Source provenance and third-party licensing checks for the expanded research code are tracked in [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md); inclusion here should not be read as a claim that every paper result has been re-run.
