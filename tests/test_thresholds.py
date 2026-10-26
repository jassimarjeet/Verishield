"""Validation-only demo thresholds. No FaceForensics download and no test videos."""

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.thresholds import (
    VALIDATION_SPLIT,
    choose_thresholds,
    load_checkpoint,
    sha256_file,
    validation_files,
    write_thresholds,
)
from src.models.xception import build_xception


def _clip(score, label):
    return {"score": score, "label": label, "split": VALIDATION_SPLIT, "video_id": "179", "method": "original"}


def test_medians_separate_pristine_and_fake():
    chosen = choose_thresholds(
        [_clip(0.20, 0), _clip(0.40, 0), _clip(0.60, 1), _clip(0.90, 1)]
    )
    assert chosen["calibration"] == "no extra calibration"
    assert chosen["selection"] == "validation-medians"
    assert chosen["tau1"] == 0.3
    assert chosen["tau2"] == 0.75
    assert chosen["false_accept_count"] == 0
    assert chosen["false_reject_count"] == 0
    assert chosen["three_outcomes"] is True
    assert chosen["n_manual_review"] == 2
    assert chosen["run_id"] == "example-xception-face-c23-regime-s"


def test_equal_pristine_scores_still_leave_one_under_tau1():
    chosen = choose_thresholds(
        [_clip(0.20, 0), _clip(0.20, 0), _clip(0.80, 1), _clip(0.90, 1)]
    )
    assert chosen["tau1"] == 0.21
    assert chosen["tau2"] == 0.85
    assert chosen["false_accept_count"] == 0
    assert chosen["false_reject_count"] == 0
    assert chosen["three_outcomes"] is True


def test_overlap_pulls_one_validation_clip_into_the_band():
    chosen = choose_thresholds(
        [_clip(0.30, 0), _clip(0.70, 0), _clip(0.40, 1), _clip(0.90, 1)]
    )
    assert chosen["tau1"] == 0.5
    assert chosen["tau2"] == 0.71
    assert chosen["false_accept_count"] == 1
    assert chosen["false_reject_count"] == 0
    assert chosen["three_outcomes"] is True


def test_inverted_scores_use_the_validation_grid_and_say_so():
    chosen = choose_thresholds([_clip(0.80, 0), _clip(0.20, 1)])
    assert chosen["selection"] == "validation-grid"
    assert chosen["tau1"] == 0.20
    assert chosen["tau2"] == 0.81
    assert chosen["three_outcomes"] is False
    assert chosen["false_accept_count"] == 0
    assert chosen["false_reject_count"] == 0


def test_test_split_cannot_choose_tau():
    try:
        choose_thresholds([{"score": 0.2, "label": 0, "split": "test"}])
    except RuntimeError as exc:
        assert "Test videos are not used" in str(exc)
    else:
        raise AssertionError("a test split was allowed to choose tau")


def test_held_out_file_list_excludes_train_and_official_test_ids():
    rows = validation_files(Path("/content/ffpp"))
    names = sorted(video_id for _path, video_id, _label, _method in rows)
    assert names == [
        "179",
        "179_212",
        "179_212",
        "179_212",
        "179_212",
        "212",
        "212_179",
        "212_179",
        "212_179",
        "212_179",
    ]
    joined = " ".join(names)
    for banned in ("000", "011", "805", "140"):
        assert banned not in joined.split()
        assert f"{banned}_" not in joined and f"_{banned}" not in joined
    text = (ROOT / "src" / "eval" / "thresholds.py").read_text()
    assert "test.txt" not in text
    assert "test.json" not in text
    assert "INDEX.md" not in text


def test_write_thresholds_requires_the_real_file_hash(tmp_path):
    weights = tmp_path / "best.pt"
    payload_bytes = b"not-a-real-network"
    weights.write_bytes(payload_bytes)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    assert sha256_file(weights) == digest
    chosen = choose_thresholds(
        [_clip(0.20, 0), _clip(0.40, 0), _clip(0.60, 1), _clip(0.90, 1)]
    )
    chosen["weight_sha256"] = "0" * 64
    chosen["checkpoint"] = str(weights)
    out = tmp_path / "thresholds-val.json"
    try:
        write_thresholds(out, chosen, weights)
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a mismatched hash was written")
    assert not out.exists()
    chosen["weight_sha256"] = digest
    write_thresholds(out, chosen, weights)
    saved = json.loads(out.read_text())
    assert saved["weight_sha256"] == digest
    assert saved["tau1"] == 0.3
    index = (ROOT / "artifacts" / "INDEX.md").read_text()
    assert digest not in index
    assert "|" in index.splitlines()[2]


def test_checkpoint_roundtrip_does_not_download_weights(tmp_path):
    model = build_xception(pretrained=False)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = load_checkpoint(path)
    left = torch.zeros(1, 3, 299, 299)
    with torch.no_grad():
        assert torch.allclose(model(left), loaded(left))


def test_notebook_cell_hashes_the_drive_checkpoint_at_runtime():
    notebook = json.loads((ROOT / "colab" / "train_xception.ipynb").read_text())
    source = "".join(notebook["cells"][-1]["source"])
    assert 'Path("/content/drive/MyDrive/verishield/checkpoints/regime-s-c23/best.pt")' in source
    assert "sha256_file" in source
    assert "score_held_out_pair" in source
    assert "choose_thresholds" in source
    assert "train_regime_s" not in source
    assert "test.json" not in source
    assert not any(len(token) == 64 and all(ch in "0123456789abcdef" for ch in token) for token in source.split())
