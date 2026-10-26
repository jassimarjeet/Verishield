"""Demo API decisions. No checkpoint file and no dataset files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.decisions import decision_payload
from src.eval.thresholds import classify_score


def _client() -> TestClient:
    return TestClient(app)


def test_threshold_decision_does_not_need_a_checkpoint_or_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    assert not (tmp_path / "missing.pt").exists()
    client = _client()

    cuts = client.get("/api/thresholds")
    assert cuts.status_code == 200
    published = cuts.json()
    assert published["tau1"] == 0.853444
    assert published["tau2"] == 0.86573
    assert published["accept"] == "score < tau1"
    assert published["manual_review"] == "tau1 <= score < tau2"
    assert published["reject"] == "score >= tau2"
    assert published["score_name"] == "mean fake-class softmax"
    assert "score" not in published

    cases = (
        (0.0, "accept"),
        (0.85, "accept"),
        (0.853444, "manual_review"),
        (0.86, "manual_review"),
        (0.86573, "reject"),
        (0.9, "reject"),
        (1.0, "reject"),
    )
    for score, expected in cases:
        response = client.post("/api/decision", json={"score": score})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["decision"] == expected
        assert body["decision"] == classify_score(body["score"], body["tau1"], body["tau2"])
        assert body["decision"] == decision_payload(score)["decision"]
        assert body["tau1"] == 0.853444
        assert body["tau2"] == 0.86573
        assert set(body["bands"]) == {"accept", "manual_review", "reject"}
        assert body["explanation"] == body["bands"][expected]

    refused = client.post("/api/decision", json={"score": 1.2})
    assert refused.status_code == 400
    assert "score" not in refused.json()


def test_detect_without_a_checkpoint_does_not_invent_a_score(monkeypatch, tmp_path):
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    client = _client()
    missing_env = client.post(
        "/api/detect",
        files={"file": ("face.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
    )
    assert missing_env.status_code == 503
    detail = missing_env.json()
    assert "score" not in detail
    assert "decision" not in detail
    assert "tau1" not in detail
    assert "CHECKPOINT_PATH" in detail["detail"]
    assert "no score" in detail["detail"].lower()

    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    missing_file = client.post(
        "/api/detect",
        files={"file": ("clip.mp4", b"not-a-video", "video/mp4")},
    )
    assert missing_file.status_code == 503
    body = missing_file.json()
    assert "score" not in body
    assert "decision" not in body
    assert "missing" in body["detail"].lower()
    assert "no score" in body["detail"].lower()
    assert not (tmp_path / "best.pt").exists()


def test_unsupported_upload_has_no_score(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    response = _client().post(
        "/api/detect",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415
    assert "score" not in response.json()
    assert "no score" in response.json()["detail"].lower()
