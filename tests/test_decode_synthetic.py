"""Decode a synthetic color clip generated at runtime.

The clip is produced by FFmpeg during the test. It is not a FaceForensics
file, and the test does not leave an mp4, avi, or png in the repository.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2

FRAME_WIDTH = 64
FRAME_HEIGHT = 48
FRAME_COUNT = 6
FRAME_RATE = 10


def _generate_color_clip(path: Path) -> None:
    source = f"testsrc=size={FRAME_WIDTH}x{FRAME_HEIGHT}:rate={FRAME_RATE}"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        source,
        "-frames:v",
        str(FRAME_COUNT),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise AssertionError(
            "ffmpeg failed to write the synthetic clip:\n"
            f"{completed.stderr}"
        )


def _read_frames(path: Path) -> list:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"OpenCV could not open {path.name}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def test_opencv_reads_synthetic_color_clip():
    with tempfile.TemporaryDirectory() as directory:
        clip = Path(directory) / "synthetic-color.mp4"
        _generate_color_clip(clip)
        frames = _read_frames(clip)

    assert len(frames) == FRAME_COUNT
    for frame in frames:
        height, width = frame.shape[:2]
        assert (width, height) == (FRAME_WIDTH, FRAME_HEIGHT)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
