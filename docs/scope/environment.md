# Environment

Working choice for this machine, recorded on 2026-10-26 17:38:54 IST (Asia/Calcutta, UTC+5:30). The clock was converted from UTC with the `Asia/Kolkata` zoneinfo entry, the current name of Asia/Calcutta. This image ships that entry and does not ship an `Asia/Calcutta` symlink.

Leave this choice in place after the first trained checkpoint. A later change needs a new experiment id.

## Virtual environment

Created with the standard-library `venv` module:

```text
python3 -m venv .venv
```

The tool is `venv`. This machine has `/usr/bin/python3` and no `python` binary on `PATH`, so the invocation is `python3 -m venv`. `.venv/` is gitignored and is not committed.

## Python

```text
3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]
```

Interpreter: `.venv/bin/python`.

Neither the FaceForensics++ paper nor the project notebook pins a Python version. Python 3.12.3 is a working choice for this machine, not a source requirement.

## PyTorch check

Command, run with `.venv/bin/python` on 2026-10-26 17:38:54 IST:

```text
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Output:

```text
2.14.1+cu130 False
```

GPU availability: false.

CUDA is false. `torch.cuda.is_available()` returned `False`, and `torch.cuda.device_count()` is 0. pip's default wheel is a CUDA 13.0 build (`2.14.1+cu130`; `torch.version.cuda` is `13.0`), and this machine has no usable GPU. Regime F is not possible on this machine.

## Packages installed at this step

`pip install` was given only these names, with no version specifiers. The roadmap does not pin them. `pyproject.toml` lists the same names with no version pins. The versions below are what this install resolved:

| Package | pip version | Imported version |
| --- | --- | --- |
| torch | 2.14.1 | 2.14.1+cu130 |
| torchvision | 0.29.1 | 0.29.1+cu130 |
| opencv-python | 5.0.0.93 | |
| numpy | 2.5.4 | |
| PyYAML | 6.0.3 | |
| pytest | 9.1.1 | |
| fastapi | 0.162.0 | |
| uvicorn | 0.54.0 | |
| httpx | 0.28.1 | |

pip also installed the dependencies of those packages, including the CUDA 13 libraries that the torch wheel requires. Those dependency versions are an observation of this install. They are not additional project requirements.

## Colab check

`colab/check_env.ipynb` has one code cell. It prints the Python version, `torch.__version__`, `torch.cuda.is_available()`, and `torchvision.__version__`. The notebook contains no dataset path.
