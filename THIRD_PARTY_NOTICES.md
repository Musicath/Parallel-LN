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
The [upstream license](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE)
is the same license included as `LICENSE.txt` in this repository.

PyTorch, torchvision, NumPy, and other installed dependencies are imported as
libraries and are not vendored in this repository. Their own licenses apply to
their distributions.
