"""Official FaceForensics++ video-id lists and the Regime S subsample.

The JSON files in ``official_splits`` are the authors' published id lists from
https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
They are not videos.
"""

import json
import random
from pathlib import Path

SPLIT_DIR = Path(__file__).resolve().parent / "official_splits"
REGIME_PAIRS = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "examples"
    / "regime-s-c23-pairs.json"
)

# Python 3.12 random.Random(0).sample of the 360 official train pairs, k=5.
# seed 0 is an example seed, not a seed published by the paper.
REGIME_SEED = 0
REGIME_PAIR_COUNT = 5
MONITOR_PAIR_COUNT = 1


def load_pairs(name: str) -> list[list[str]]:
    payload = json.loads((SPLIT_DIR / name).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{name} is not a list of pairs")
    return payload


def pair_ids(pairs: list[list[str]]) -> set[str]:
    found = set()
    for pair in pairs:
        found.update(pair)
    return found


def official_ids() -> dict[str, set[str]]:
    return {
        "train": pair_ids(load_pairs("train.json")),
        "val": pair_ids(load_pairs("val.json")),
        "test": pair_ids(load_pairs("test.json")),
    }


def draw_regime_pairs(train_pairs: list[list[str]] | None = None) -> list[list[str]]:
    pairs = list(train_pairs) if train_pairs is not None else load_pairs("train.json")
    drawn = random.Random(REGIME_SEED).sample(pairs, REGIME_PAIR_COUNT)
    return [list(pair) for pair in drawn]


def load_frozen_regime() -> dict[str, list[list[str]]]:
    payload = json.loads(REGIME_PAIRS.read_text())
    train_pairs = payload["train_pairs"]
    monitor_pairs = payload["monitor_pairs"]
    banned = official_ids()
    for label, pairs in ("train", train_pairs), ("monitor", monitor_pairs):
        ids = pair_ids(pairs)
        if ids & banned["val"] or ids & banned["test"]:
            raise ValueError(f"{label} pairs include an official validation or test id")
        if not ids <= banned["train"]:
            raise ValueError(f"{label} pairs are not inside the official training ids")
    return {"train_pairs": train_pairs, "monitor_pairs": monitor_pairs}


def assert_train_ids_only(ids: set[str]) -> None:
    banned = official_ids()
    if ids & banned["val"] or ids & banned["test"]:
        raise ValueError("official validation or test ids are not allowed in this run")
    if not ids <= banned["train"]:
        raise ValueError("every Regime S id must come from the official training list")
