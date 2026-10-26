"""Regime S split, crop factor, and a one-batch smoke test.

The smoke test does not download FaceForensics++ and does not write a checkpoint.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ffpp_download import (
    V1_URL,
    V4_URL,
    select_regime_filenames,
    stems_for_dataset,
    v4_argv,
)
from src.data.ffpp_splits import (
    REGIME_PAIR_COUNT,
    REGIME_SEED,
    draw_regime_pairs,
    load_frozen_regime,
    load_pairs,
    official_ids,
    pair_ids,
)
from src.models.xception import build_xception, freeze_backbone
from src.pipeline.crop import CROP_FACTOR, crop_largest_face, enlarge_box


def test_regime_s_ids_stay_inside_official_train():
    frozen = load_frozen_regime()
    drawn = draw_regime_pairs()
    assert drawn[:4] == frozen["train_pairs"]
    assert drawn[4:] == frozen["monitor_pairs"]
    assert random.Random(REGIME_SEED).sample(load_pairs("train.json"), REGIME_PAIR_COUNT) == [
        pair for group in (frozen["train_pairs"], frozen["monitor_pairs"]) for pair in group
    ]
    ids = official_ids()
    used = pair_ids(frozen["train_pairs"] + frozen["monitor_pairs"])
    assert used <= ids["train"]
    assert not used & ids["val"]
    assert not used & ids["test"]
    assert "000" in ids["test"]
    assert "000" not in used
    assert len(ids["train"]) == 720
    assert len(ids["val"]) == 140
    assert len(ids["test"]) == 140


def test_download_refuses_youtube_and_keeps_pair_names():
    with pytest.raises(SystemExit):
        stems_for_dataset("original_youtube_videos", [["011", "805"]])
    with pytest.raises(SystemExit):
        v4_argv(Path("faceforensics_download_v4.py"), Path("/content/ffpp"), "all", "EU")
    argv = v4_argv(Path("faceforensics_download_v4.py"), Path("/content/ffpp"), "Deepfakes", "EU")
    assert argv[argv.index("-c") + 1] == "c23"
    assert argv[argv.index("-t") + 1] == "videos"
    assert argv[argv.index("-d") + 1] == "Deepfakes"
    assert "--server" in argv
    assert "raw" not in argv
    assert "c40" not in argv
    assert "original_youtube_videos" not in argv
    assert V4_URL.endswith("faceforensics_download_v4.py")
    assert V1_URL.endswith("faceforensics_download_v1.py")
    assert stems_for_dataset("original", [["011", "805"]]) == ["011", "805"]
    assert stems_for_dataset("Deepfakes", [["011", "805"]]) == ["011_805", "805_011"]
    kept = select_regime_filenames(["000.mp4", "011.mp4", "011_805.mp4"], {"011.mp4", "011_805.mp4"})
    assert kept == ["011.mp4", "011_805.mp4"]


def test_crop_factor_is_1_3_and_clips_at_the_border():
    assert CROP_FACTOR == 1.3
    enlarged = enlarge_box(100, 80, 40, 20, 400, 400)
    x0, y0, x1, y1 = enlarged["unclipped"]
    assert abs((x1 - x0) - 40 * 1.3) <= 1
    assert abs((y1 - y0) - 20 * 1.3) <= 1
    clipped = enlarge_box(0, 0, 10, 10, 30, 30)
    assert clipped["crop_box"][0] == 0
    assert clipped["crop_box"][1] == 0
    assert clipped["unclipped"][0] < 0


def test_missed_face_is_not_replaced_by_the_full_frame():
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    found = crop_largest_face(blank)
    assert found["ok"] is False
    assert found["crop_image"] is None


def test_training_code_does_not_name_test_txt():
    text = (ROOT / "src" / "train" / "regime_s.py").read_text()
    assert "test.txt" not in text


def test_xception_one_batch_smoke():
    """Smoke test only: random tensors, no ImageNet download, no checkpoint."""
    model = build_xception(pretrained=False)
    freeze_backbone(model)
    classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    for parameter in model.parameters():
        assert parameter.requires_grad is (id(parameter) in classifier_ids)
    images = torch.randn(2, 3, 299, 299)
    labels = torch.tensor([0, 1])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.0002,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    assert logits.shape == (2, 2)
    assert torch.isfinite(loss)
    index = (ROOT / "artifacts" / "INDEX.md").read_text()
    assert "legacy_xception" not in index
