"""One-cell CIFAR-10 VGG-16 run; reconstructed from the archived grid.

This is a portable training entry point, not recovered numerical evidence for
the paper's 945-run figure. See README.md here before interpreting results.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.vgg.model import ACTIVATIONS, PaperVGG16


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
LR_DECAY_EPOCHS = (50, 90, 130, 170)  # one-based, applied before these epochs
LR_DECAY_FACTOR = 1 / 2.5


def learning_rate_for_epoch(initial_lr, epoch_number, warmup_epochs=10):
    """Paper schedule using one-based epoch numbers, without scheduler state.

    Epochs 1..10 linearly warm up to the selected initial rate. The first
    decay is applied before training epoch 50, then at 90, 130 and 170.
    """
    if epoch_number < 1 or warmup_epochs < 0:
        raise ValueError("epoch_number must be positive and warmup nonnegative")
    if warmup_epochs and epoch_number <= warmup_epochs:
        return initial_lr * epoch_number / warmup_epochs
    decays = sum(epoch_number >= milestone for milestone in LR_DECAY_EPOCHS)
    return initial_lr * LR_DECAY_FACTOR ** decays


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for this single run; existing files are never overwritten")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation", choices=ACTIVATIONS, default="pln")
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="development smoke test only; invalidates paper comparison")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers cannot be negative")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("warmup epochs must be between zero and total epochs")
    if args.lr <= 0 or args.init_scale <= 0 or args.weight_decay < 0:
        parser.error("learning rate and scale must be positive; weight decay cannot be negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    return args


def make_loaders(args):
    common = [transforms.Resize((32, 32)), transforms.ToTensor(),
              transforms.Normalize(MEAN, STD)]
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train = datasets.CIFAR10(args.data_root, train=True, transform=train_transform,
                             download=args.download)
    test = datasets.CIFAR10(args.data_root, train=False,
                            transform=transforms.Compose(common),
                            download=args.download)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers,
                              pin_memory=args.device.startswith("cuda"))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers,
                             pin_memory=args.device.startswith("cuda"))
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    correct = total = 0
    loss_sum = 0.0
    for batch_no, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
        if max_batches is not None and batch_no >= max_batches:
            break
    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total, loss_sum / total


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty output directory: {args.output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True  # historical trainer setting
    device = torch.device(args.device)
    train_loader, test_loader = make_loaders(args)
    model = PaperVGG16(args.activation, args.init_scale).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(torch_version=torch.__version__,
                  torchvision_version=__import__("torchvision").__version__,
                  lr_decay_epochs=LR_DECAY_EPOCHS,
                  lr_decay_factor=LR_DECAY_FACTOR)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                             encoding="utf-8")
    best_test_accuracy = 0.0
    with (args.output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            epoch_number = epoch + 1
            lr = learning_rate_for_epoch(args.lr, epoch_number, args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            current_lr = optimizer.param_groups[0]["lr"]
            train_accuracy, train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                max_batches=args.max_train_batches)
            test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
            best_test_accuracy = max(best_test_accuracy, test_accuracy)
            record = dict(epoch=epoch, epoch_number=epoch_number, lr=current_lr,
                          train_accuracy=train_accuracy, train_loss=train_loss,
                          test_accuracy=test_accuracy, test_loss=test_loss,
                          best_test_accuracy=best_test_accuracy)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    run(parse_args())
