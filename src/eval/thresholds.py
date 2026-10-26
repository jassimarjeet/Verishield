"""Pick the demo thresholds on Regime S validation clips only.

The paper's accuracy stays a 0.5 cut on the fake-class softmax. This module
picks the two demo cuts, tau1 and tau2. The probability is that same softmax.
No second calibration is fit.

A clip score is the mean fake-class softmax of its cropped frames. Validation
clips are the held-out official training pair only. Official test videos are
not read, and the videos the optimizer fit are not used to move tau.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from src.data.ffpp_splits import assert_train_ids_only, load_frozen_regime, pair_ids
from src.models.xception import build_xception
from src.pipeline.crop import crop_largest_face
from src.pipeline.decode import iter_bgr_frames
from src.train.regime_s import MAX_CROPS_PER_VIDEO, _files_for_pairs, _rgb_tensor

CALIBRATION = "no extra calibration"
VALIDATION_SPLIT = "regime-s-monitor"
RUN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "examples"
    / "xception-face-c23-regime-s.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regime_s_run_id() -> str:
    payload = yaml.safe_load(RUN_CONFIG.read_text())
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError(f"{RUN_CONFIG} has no run_id.")
    return run_id


def validation_files(data_root: Path) -> list[tuple[Path, str, int, str]]:
    """c23 files for the held-out pair. Does not open the videos."""
    regime = load_frozen_regime()
    monitor_ids = pair_ids(regime["monitor_pairs"])
    train_ids = pair_ids(regime["train_pairs"])
    rows = _files_for_pairs(Path(data_root), regime["monitor_pairs"])
    if not rows:
        raise RuntimeError("The held-out pair has no files. Not choosing a threshold.")
    for _path, video_id, label, _method in rows:
        stem_ids = set(str(video_id).split("_"))
        if not stem_ids <= monitor_ids:
            raise RuntimeError(f"{video_id} is outside the held-out pair.")
        if stem_ids & train_ids:
            raise RuntimeError("Videos the optimizer fit cannot set tau.")
        if label not in (0, 1):
            raise RuntimeError(f"Unexpected label on {video_id}.")
        assert_train_ids_only(stem_ids)
    return rows


def load_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}. Not scoring and not inventing a hash.")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "model" not in blob:
        raise RuntimeError("Checkpoint has no model weights. Not scoring.")
    model = build_xception(pretrained=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def score_held_out_pair(checkpoint_path: Path, data_root: Path) -> list[dict]:
    """Score the held-out pair. A clip with no face crop gets no invented score."""
    model = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    clips = []
    for path, video_id, label, method in validation_files(data_root):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing validation file {path}. Not inventing a score."
            )
        frames = []
        for _index, frame in iter_bgr_frames(path):
            found = crop_largest_face(frame)
            if not found["ok"]:
                continue
            frames.append(_rgb_tensor(found["crop_image"]))
            if len(frames) >= MAX_CROPS_PER_VIDEO:
                break
        if not frames:
            print(f"No face crop in {path}. That clip has no score and does not move tau.")
            continue
        batch = torch.stack(frames).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(batch), dim=1)[:, 1]
        clips.append(
            {
                "video_id": video_id,
                "method": method,
                "label": int(label),
                "score": float(probabilities.mean().item()),
                "n_frames": len(frames),
                "split": VALIDATION_SPLIT,
            }
        )
    return clips


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _hundredth_above(value: float) -> float | None:
    step = math.floor(value * 100 + 1e-9) + 1
    if step >= 100:
        return None
    candidate = step / 100
    if candidate <= value:
        return None
    return round(candidate, 2)


def _classify(score: float, tau1: float, tau2: float) -> str:
    if score < tau1:
        return "accept"
    if score < tau2:
        return "manual_review"
    return "reject"


def _tradeoff(clips: list[dict], tau1: float, tau2: float) -> dict:
    decisions = [_classify(clip["score"], tau1, tau2) for clip in clips]
    false_accept = 0
    false_reject = 0
    pristine_accept = 0
    fake_reject = 0
    for clip, decision in zip(clips, decisions):
        if clip["label"] == 1 and decision == "accept":
            false_accept += 1
        if clip["label"] == 0 and decision == "reject":
            false_reject += 1
        if clip["label"] == 0 and decision == "accept":
            pristine_accept += 1
        if clip["label"] == 1 and decision == "reject":
            fake_reject += 1
    review = sum(decision == "manual_review" for decision in decisions)
    return {
        "decisions": decisions,
        "false_accept_count": false_accept,
        "false_reject_count": false_reject,
        "n_accept": sum(decision == "accept" for decision in decisions),
        "n_manual_review": review,
        "n_reject": sum(decision == "reject" for decision in decisions),
        "pristine_accept": pristine_accept,
        "fake_reject": fake_reject,
        "three_outcomes": pristine_accept >= 1 and fake_reject >= 1 and review >= 1,
    }


def _ends_hold(clips: list[dict], tau1: float, tau2: float) -> bool:
    if not tau1 < tau2:
        return False
    stats = _tradeoff(clips, tau1, tau2)
    return stats["pristine_accept"] >= 1 and stats["fake_reject"] >= 1


def _from_grid(clips: list[dict]) -> tuple[float, float]:
    best = None
    best_key = None
    for left in range(1, 100):
        for right in range(left + 1, 100):
            tau1 = round(left / 100, 2)
            tau2 = round(right / 100, 2)
            stats = _tradeoff(clips, tau1, tau2)
            key = (
                0 if stats["three_outcomes"] else 1,
                stats["false_accept_count"],
                stats["false_reject_count"],
                stats["n_manual_review"] if stats["three_outcomes"] else 0,
                round(tau2 - tau1, 2),
                tau1,
                tau2,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (tau1, tau2)
    if best is None:
        raise RuntimeError("No threshold pair could be scored. Not inventing tau.")
    return best


def _pull_into_band(clips: list[dict], tau1: float, tau2: float) -> tuple[float, float]:
    if _tradeoff(clips, tau1, tau2)["n_manual_review"] > 0:
        return tau1, tau2
    midpoint = (tau1 + tau2) / 2
    order = sorted(clips, key=lambda clip: (abs(clip["score"] - midpoint), clip["score"], clip["label"]))
    for clip in order:
        score = clip["score"]
        new1, new2 = tau1, tau2
        if tau1 <= score < tau2:
            return tau1, tau2
        if score < tau1:
            new1 = round(score, 6)
        else:
            bumped = _hundredth_above(score)
            if bumped is None:
                continue
            new2 = bumped
        if _ends_hold(clips, new1, new2) and _tradeoff(clips, new1, new2)["n_manual_review"] > 0:
            return new1, new2
    return tau1, tau2


def _from_medians(clips: list[dict], pristine: list[float], fake: list[float]) -> tuple[float, float] | None:
    tau1 = round(_median(pristine), 6)
    tau2 = round(_median(fake), 6)
    if not (tau1 < tau2 and any(score < tau1 for score in pristine)):
        bumped = _hundredth_above(min(pristine))
        if bumped is None or not bumped < tau2:
            return None
        tau1 = bumped
    if not any(score >= tau2 for score in fake):
        higher = sorted(score for score in fake if score > tau1)
        if not higher:
            return None
        tau2 = round(higher[0], 6)
    if not _ends_hold(clips, tau1, tau2):
        return None
    return _pull_into_band(clips, tau1, tau2)


def choose_thresholds(clips: list[dict]) -> dict:
    """Choose tau1 and tau2 from these validation clips and no other scores."""
    if not clips:
        raise RuntimeError("No validation scores. Not choosing a threshold.")
    prepared = []
    for clip in clips:
        if clip.get("split") != VALIDATION_SPLIT:
            raise RuntimeError(
                "Thresholds are chosen on the Regime S held-out pair only. Test videos are not used."
            )
        label = clip.get("label")
        if label not in (0, 1):
            raise RuntimeError("Each validation clip needs label 0 (pristine) or 1 (fake).")
        score = clip.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise RuntimeError("Each validation clip needs a numeric score.")
        if not 0.0 <= float(score) <= 1.0:
            raise RuntimeError("A fake-class softmax must sit in [0, 1]. Not choosing a threshold.")
        prepared.append({**clip, "label": int(label), "score": round(float(score), 6)})
    pristine = [clip["score"] for clip in prepared if clip["label"] == 0]
    fake = [clip["score"] for clip in prepared if clip["label"] == 1]
    if not pristine or not fake:
        raise RuntimeError("Validation needs both a pristine clip and a fake clip. Not choosing a threshold.")
    chosen = _from_medians(prepared, pristine, fake)
    selection = "validation-medians"
    if chosen is None:
        chosen = _from_grid(prepared)
        selection = "validation-grid"
    tau1, tau2 = chosen
    stats = _tradeoff(prepared, tau1, tau2)
    n_pristine = len(pristine)
    n_fake = len(fake)
    scored = []
    for clip, decision in zip(prepared, stats["decisions"]):
        scored.append(
            {
                "video_id": clip.get("video_id"),
                "method": clip.get("method"),
                "label": clip["label"],
                "score": clip["score"],
                "n_frames": clip.get("n_frames"),
                "split": VALIDATION_SPLIT,
                "decision": decision,
            }
        )
    return {
        "run_id": regime_s_run_id(),
        "calibration": CALIBRATION,
        "score": "mean fake-class softmax",
        "validation": "held-out official training pair 179/212",
        "selection": selection,
        "tau1": tau1,
        "tau2": tau2,
        "false_accept_count": stats["false_accept_count"],
        "false_reject_count": stats["false_reject_count"],
        "false_accept_rate": stats["false_accept_count"] / n_fake,
        "false_reject_rate": stats["false_reject_count"] / n_pristine,
        "n_pristine": n_pristine,
        "n_fake": n_fake,
        "n_accept": stats["n_accept"],
        "n_manual_review": stats["n_manual_review"],
        "n_reject": stats["n_reject"],
        "three_outcomes": stats["three_outcomes"],
        "clips": scored,
    }


def write_thresholds(path: Path, payload: dict, checkpoint_path: Path) -> None:
    """Write the frozen threshold file only when the hash matches the weights."""
    checkpoint_path = Path(checkpoint_path)
    digest = sha256_file(checkpoint_path)
    claimed = payload.get("weight_sha256")
    if claimed != digest:
        raise RuntimeError(
            "Refusing to write thresholds whose weight hash does not match the checkpoint file."
        )
    if payload.get("calibration") != CALIBRATION:
        raise RuntimeError("The probability is the two-class softmax. Not writing a second calibration.")
    if payload.get("run_id") != regime_s_run_id():
        raise RuntimeError("The threshold file run_id does not match the Regime S config.")
    for key in ("tau1", "tau2", "false_accept_count", "false_reject_count"):
        if payload.get(key) is None:
            raise RuntimeError(f"Missing {key}. Not writing an empty threshold file.")
    if not payload["tau1"] < payload["tau2"]:
        raise RuntimeError("tau1 must be below tau2.")
    for clip in payload.get("clips", []):
        if clip.get("split") != VALIDATION_SPLIT:
            raise RuntimeError("Not writing thresholds that include a non-validation clip.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
