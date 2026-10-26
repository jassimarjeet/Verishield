"""Read video frames in OpenCV's BGR order.

Frame index 0 is the first decodable frame. Later stages convert a crop to
RGB before the network.
"""

from pathlib import Path

import cv2


def iter_bgr_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()
