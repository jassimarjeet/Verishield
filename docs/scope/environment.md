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

## Compute roles

Jas confirmed on 2026-10-26 (Asia/Calcutta, IST): training runs on Google Colab. This checkout stays the test and app machine. Recorded in this file on 2026-10-26 17:47:13 IST.

The Python 3.12.3 and torch 2.14.1+cu130 record above stays as written. `.venv` from that step was still present and was reused. Packages were not reinstalled, so those recorded versions are unchanged.

## Media tools

Checked on this machine on 2026-10-26 17:47:13 IST. Both commands succeeded.

`ffmpeg -version` version line:

```text
ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers
```

`ffprobe -version` version line:

```text
ffprobe version 6.1.1-3ubuntu5 Copyright (c) 2007-2023 the FFmpeg developers
```

`tests/test_decode_synthetic.py` generates a short synthetic color clip with FFmpeg at runtime and checks that OpenCV reads the same clip at the known frame count and frame size. The clip is not a FaceForensics file and is not committed.

The same `ffmpeg -version` and `ffprobe -version` check inside the future container is still open. That check belongs to Step 3.8. This step does not build the API image.

## Storage

Working paths for this project, recorded on 2026-10-26 17:52:20 IST (Asia/Calcutta):

- Dataset root: `data/ffpp/`. Gitignored. The directory exists on disk. Nothing under it is committed.
- Checkpoint root: `artifacts/checkpoints/`. Gitignored. The directory exists on disk. Nothing under it is committed.
- `artifacts/INDEX.md` is tracked. It has a header and no checkpoint rows yet. The SHA-256 of each weight file is filled after each run. The weight file stays out of git.

Training runs on Google Colab. A Colab ephemeral disk is not the backup.

Backup rule: at least one copy of the best checkpoint of each final run goes on a second disk Jas controls. The physical disk is still unnamed. No disk path is recorded.

## Tracker deviation

Recorded on 2026-10-26 18:39:17 IST (Asia/Calcutta).

The paper's tracker is the Face2Face tracker of Thies et al. That tracker is not installed on this machine, and the public FaceForensics repository does not ship it as a pip package. opencv-python 5.0.0 also has no Haar cascade. The Colab Regime S notebook uses OpenCV FaceDetectorYN with the YuNet 2023mar ONNX from `opencv/opencv_zoo` (`face_detection_yunet_2023mar.onnx`, score threshold 0.9). The crop is still enlarged by 1.3 about the detected box center. A frame with no detection is skipped and is not replaced by the full frame. The ONNX file is downloaded at runtime and is not committed.

## Download scripts

Recorded on 2026-10-26 18:54:57 IST (Asia/Calcutta).

The training download is FaceForensics++ script v4: `https://kaldir.vc.in.tum.de/faceforensics_download_v4.py`. The notebook fetches that file into `/content/` and calls it. Its flags for this run are the output directory, `-d` one of `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, or `NeuralTextures`, `-c c23`, `-t videos`, and `--server EU`. v4's `main` prints its terms URL and waits for Enter. That prompt is left in place.

The original FaceForensics script is v1: `https://kaldir.vc.in.tum.de/faceforensics_download_v1.py`. The training download does not call v1.

## Regime S notebook

`colab/train_xception.ipynb` is the notebook Jas pastes into the empty Colab file. torchvision 0.29.1 has no Xception class, so the notebook installs timm and loads `legacy_xception`. On this machine `pip` installed timm 1.0.30 for a one-batch smoke test with `pretrained=False`. That smoke test does not download FaceForensics++ weights or videos, and it does not add a row to `artifacts/INDEX.md`. ImageNet initialization happens when the Colab notebook runs with `pretrained=True`.
