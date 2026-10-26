"""Score one uploaded image or short video with the study checkpoint.

The clip score matches the validation scorer: the mean fake-class softmax of
up to 16 enlarged face crops. A missing checkpoint, an unreadable file, or a
missed face does not become a score.
"""

import os
from pathlib import Path

import cv2
import torch

from src.api.errors import DetectionError
from src.eval.thresholds import load_checkpoint
from src.pipeline.crop import crop_largest_face
from src.pipeline.decode import iter_bgr_frames
from src.train.regime_s import MAX_CROPS_PER_VIDEO, _rgb_tensor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
MAX_VIDEO_SECONDS = 15
MAX_SCAN_FRAMES = 90
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 40 * 1024 * 1024

_MODEL_CACHE: dict = {"path": None, "mtime_ns": None, "model": None, "device": None}


def media_kind(filename: str, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in IMAGE_TYPES:
        return "image"
    if ctype in VIDEO_TYPES:
        return "video"
    raise DetectionError(
        "Upload an image (JPEG, PNG, or WebP) or a short video (MP4, MOV, or WebM). No score was produced.",
        415,
    )


def size_limit(kind: str) -> int:
    return MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES


def resolve_checkpoint() -> Path:
    raw = os.environ.get("CHECKPOINT_PATH", "").strip()
    if not raw:
        raise DetectionError(
            "CHECKPOINT_PATH is not set. Detection did not run and no score was produced.",
            503,
        )
    path = Path(raw)
    if not path.is_file():
        raise DetectionError(
            f"Checkpoint file is missing at {path}. Detection did not run and no score was produced.",
            503,
        )
    return path


def _cached_model(checkpoint: Path):
    mtime_ns = checkpoint.stat().st_mtime_ns
    if (
        _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["path"] == checkpoint
        and _MODEL_CACHE["mtime_ns"] == mtime_ns
    ):
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"]
    model = load_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _MODEL_CACHE.update(path=checkpoint, mtime_ns=mtime_ns, model=model, device=device)
    return model, device


def _reject_long_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise DetectionError(
            "OpenCV could not open that video. No score was produced.",
            400,
        )
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    if fps > 1.0 and count > 0.0 and (count / fps) > MAX_VIDEO_SECONDS:
        raise DetectionError(
            f"This demo reads a short video of at most {MAX_VIDEO_SECONDS} seconds. No score was produced.",
            400,
        )


def _iter_frames(path: Path, kind: str):
    if kind == "image":
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise DetectionError(
                "OpenCV could not read that image. No score was produced.",
                400,
            )
        yield frame
        return
    _reject_long_video(path)
    scanned = 0
    try:
        frames = iter_bgr_frames(path)
    except RuntimeError as exc:
        raise DetectionError(
            "OpenCV could not open that video. No score was produced.",
            400,
        ) from exc
    for _index, frame in frames:
        scanned += 1
        if scanned > MAX_SCAN_FRAMES:
            break
        yield frame


def score_media(path: Path, kind: str, checkpoint: Path) -> dict:
    """Return the mean fake-class softmax for face crops in this file."""
    crops = []
    seen = 0
    for frame in _iter_frames(path, kind):
        seen += 1
        found = crop_largest_face(frame)
        if not found["ok"]:
            continue
        crops.append(_rgb_tensor(found["crop_image"]))
        if len(crops) >= MAX_CROPS_PER_VIDEO:
            break
    if not crops:
        if seen == 0:
            raise DetectionError(
                "No frames could be read from that file. No score was produced.",
                400,
            )
        raise DetectionError(
            "No face crop was found. This demo does not invent a score when the face detector misses.",
            422,
        )
    model, device = _cached_model(checkpoint)
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[:, 1]
    return {
        "score": float(probabilities.mean().item()),
        "n_frames": len(crops),
        "media_kind": kind,
    }
