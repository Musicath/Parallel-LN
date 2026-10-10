# Environment setup

Run the commands below from the repository root using **Python 3.10**, the
version on which the reference smoke-check stack was tested. The script syntax
requires Python 3.10 or newer, but the pinned PyTorch 2.4.0 wheel may not be
available for every newer Python version. Install PyTorch and torchvision as a
**matched pair** for the target CPU/CUDA platform. The pair below is the one
used for the successful eight-RTX-4090 host smoke checks, not a claim that it
is the original environment of every historical experiment.

Linux shell, CUDA 12.1:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirements-smoke.txt
python -m pip check
```

On a machine where `python3.10 -m venv` reports that `ensurepip` is missing,
install the operating system's Python 3.10 venv package or use an isolated
Python 3.10 distribution that provides venv.

Windows PowerShell, CUDA 12.1 (no shell activation required):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install -r requirements-smoke.txt
.\.venv\Scripts\python.exe -m pip check
```

For CPU-only checks, use the PyTorch `cpu` wheel index in place of `cu121`
and pin `torch==2.4.0+cpu torchvision==0.19.0+cpu`. Keep PyPI as the
extra index for indirect dependencies; this was needed by a fresh Linux
Python 3.10 installation on 2026-10-10. On Windows, first verify that
`python -c "import torch"` works using the venv's Python executable:
a fresh Python 3.11 CPU environment on one development machine instead
failed while loading PyTorch's `fbgemm.dll` or one of its dependencies.
That platform issue was not resolved by this repository's code checks; do
not use DLL downloads from unofficial sites as an installation workaround.
`requirements-smoke.txt` covers only the two platform-independent packages;
it is not a historical lockfile or a substitute for the matched PyTorch pair.
For another GPU or operating system, use the matching pair and wheel selector
in the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
or the [2.4.0/0.19.0 archive](https://pytorch.org/get-started/previous-versions/).
The manuscript's historical ViT/CNN runs used PyTorch 2.3.0+cu121 and
torchvision 0.18.0+cu121; Dense Fourier used PyTorch 2.6.0+cu124. Those
historical stacks are recorded in [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
because using the tested smoke-check environment does not establish exact
bitwise reproduction of the old runs.

Check imports and the bundled evidence without training:

```sh
python -c "import torch, torchvision, numpy, matplotlib; print(torch.__version__, torchvision.__version__)"
python -m unittest discover -s tests -v
python figures/verify_data.py
python figures/verify_fourier_data.py
python figures/plot_from_csv.py
python figures/plot_normalization_domain_schematic.py
```

The plotting commands write to ignored `figures/generated/`. CIFAR-10 images
are fetched only when a training/data-preparation command is passed
`--download`; the repository does not bundle them. Training outputs should go
under ignored `outputs/`, not into tracked source or figure-data directories.
The ViT label-artifact workflow is in [its guide](../experiments/vit/README.md).
