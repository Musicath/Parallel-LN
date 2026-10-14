# Third-party code attribution

The standalone ViT implementation in `experiments/vit/model.py` was modified
from the authors' historical `Musicath-base` model. That historical model
follows the classic Vision Transformer implementation in
[`timm`](https://github.com/huggingface/pytorch-image-models), including a
truncated-normal initialization pattern. We retain this attribution because
the historical source did not document its complete upstream lineage.

The relevant upstream ViT implementation is by Ross Wightman (copyright 2020)
and is available under the Apache License, Version 2.0. Our extracted file was
substantially reduced and modified for the paper's depth-4 CIFAR-10 experiment.
The upstream source is
[`timm/models/vision_transformer.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py).
The exact upstream revision underlying the authors' historical private copy has
not been established; this attribution is intentionally conservative and does
not claim byte-for-byte identity with the current upstream file.
The [upstream license](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE)
is the same license included as `LICENSE.txt` in this repository.

The authors' historical VGG source also shares the VGG-D configuration and
convolution-initialization pattern found in
[`torchvision/models/vgg.py`](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py).
The paper-specific extraction in `experiments/vgg/model.py` has a different
classifier and selectable normalization/activation layers. The exact lineage
and revision of the private historical source are not established, so this is
a conservative attribution rather than a claim of byte-for-byte copying.
The separate `experiments/vgg_bias_init/model.py` likewise retains a VGG-D
feature layout from the authors' historical `Musicath-base` source while
reducing its framework-dependent implementation.
For any incorporated torchvision-derived portions, retain the upstream
[BSD 3-Clause license and copyright notice](licenses/torchvision-BSD-3-Clause.txt).

PyTorch, torchvision, NumPy, and other installed dependencies are imported as
libraries and are not vendored in this repository. Their own licenses apply to
their distributions.
