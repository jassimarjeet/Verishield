"""Apply the frozen validation cuts.

The cuts are read from configs/thresholds-val.json. Accept is strictly below
tau1, manual review runs until tau2, and reject starts at tau2. The score,
when one exists, is the mean fake-class softmax. This module does not load a
checkpoint and does not read dataset files.
"""

import json
from pathlib import Path

from src.api.errors import DetectionError
from src.eval.thresholds import classify_score

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = ROOT / "configs" / "thresholds-val.json"


def _format_tau(value: float) -> str:
    return format(value, ".10f").rstrip("0").rstrip(".")


def load_cuts(path: Path | None = None) -> tuple[float, float, dict]:
    path = Path(path or THRESHOLDS_PATH)
    if not path.is_file():
        raise DetectionError(
            f"Threshold file is missing at {path}. Not inventing tau1 or tau2.",
            500,
        )
    payload = json.loads(path.read_text())
    tau1 = payload.get("tau1")
    tau2 = payload.get("tau2")
    if isinstance(tau1, bool) or isinstance(tau2, bool):
        raise DetectionError("tau1 and tau2 must be numeric. Not inventing cuts.", 500)
    if not isinstance(tau1, (int, float)) or not isinstance(tau2, (int, float)):
        raise DetectionError("Threshold file is missing tau1 or tau2.", 500)
    if not float(tau1) < float(tau2):
        raise DetectionError("tau1 must be below tau2.", 500)
    return float(tau1), float(tau2), payload


def band_explanations(tau1: float, tau2: float) -> dict[str, str]:
    low = _format_tau(tau1)
    high = _format_tau(tau2)
    return {
        "accept": (
            f"Accept means the mean fake-class softmax is below τ1 ({low}). "
            "That band stays under the lower validation cut."
        ),
        "manual_review": (
            f"Manual review means the score is from τ1 ({low}) up to, but not including, τ2 ({high}). "
            "The face score is too high to accept and too low to reject on the cut alone."
        ),
        "reject": (
            f"Reject means the score is at or above τ2 ({high}). "
            "The upper validation cut treats that band as a fake-class reading."
        ),
    }


def decision_payload(score: float) -> dict:
    """Classify a caller-supplied score. Does not open a checkpoint."""
    tau1, tau2, payload = load_cuts()
    try:
        decision = classify_score(score, tau1, tau2)
    except RuntimeError as exc:
        raise DetectionError(str(exc), 400) from exc
    bands = band_explanations(tau1, tau2)
    return {
        "score": float(score),
        "tau1": tau1,
        "tau2": tau2,
        "decision": decision,
        "explanation": bands[decision],
        "bands": bands,
        "score_name": payload.get("score") or "mean fake-class softmax",
        "calibration": payload.get("calibration") or "no extra calibration",
    }


def threshold_payload() -> dict:
    """Publish the frozen cuts. This is not a detection score."""
    tau1, tau2, payload = load_cuts()
    return {
        "tau1": tau1,
        "tau2": tau2,
        "score_name": payload.get("score") or "mean fake-class softmax",
        "calibration": payload.get("calibration") or "no extra calibration",
        "accept": "score < tau1",
        "manual_review": "tau1 <= score < tau2",
        "reject": "score >= tau2",
        "bands": band_explanations(tau1, tau2),
    }
