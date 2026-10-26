"""Face crop enlarged by 1.3.

The paper uses the Face2Face tracker from Thies et al. That tracker is not
installed. opencv-python 5.0 has no Haar cascade. The substitute is OpenCV
FaceDetectorYN with the YuNet 2023mar ONNX from opencv/opencv_zoo. A missed
detection does not become a full-frame crop.
"""

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CROP_FACTOR = 1.3
TRACKER_NAME = "opencv-yunet-2023mar"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SCORE_THRESHOLD = 0.9
_DETECTORS = {}


def yunet_path() -> Path:
    path = Path(os.environ.get("VERISHIELD_YUNET", "/tmp/face_detection_yunet_2023mar.onnx"))
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


def enlarge_box(x: float, y: float, w: float, h: float, frame_w: int, frame_h: int, factor: float = CROP_FACTOR):
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    width = w * factor
    height = h * factor
    x0 = int(round(center_x - width / 2.0))
    y0 = int(round(center_y - height / 2.0))
    x1 = int(round(center_x + width / 2.0))
    y1 = int(round(center_y + height / 2.0))
    clipped = (
        max(0, x0),
        max(0, y0),
        min(frame_w, x1),
        min(frame_h, y1),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return {
        "center_xy": (center_x, center_y),
        "crop_box": clipped,
        "unclipped": (x0, y0, x1, y1),
    }


def _detector(frame_w: int, frame_h: int):
    key = (frame_w, frame_h)
    detector = _DETECTORS.get(key)
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path()),
            "",
            (frame_w, frame_h),
            YUNET_SCORE_THRESHOLD,
            0.3,
            5000,
        )
        _DETECTORS[key] = detector
    else:
        detector.setInputSize((frame_w, frame_h))
    return detector


def crop_largest_face(frame_bgr: np.ndarray) -> dict:
    height, width = frame_bgr.shape[:2]
    _retval, faces = _detector(width, height).detect(frame_bgr)
    empty = {"ok": False, "center_xy": None, "crop_box": None, "crop_image": None, "tracker": TRACKER_NAME}
    if faces is None or len(faces) == 0:
        return empty
    areas = faces[:, 2] * faces[:, 3]
    x, y, w, h = faces[int(np.argmax(areas)), :4]
    enlarged = enlarge_box(float(x), float(y), float(w), float(h), width, height)
    if enlarged is None:
        return empty
    x0, y0, x1, y1 = enlarged["crop_box"]
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return empty
    return {
        "ok": True,
        "center_xy": enlarged["center_xy"],
        "crop_box": enlarged["crop_box"],
        "crop_image": crop,
        "tracker": TRACKER_NAME,
    }
