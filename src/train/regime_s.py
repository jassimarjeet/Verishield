"""Shortened Regime S face-crop Xception session.

This is not the paper's 3 + 15 epoch schedule and not Table 1. Official
validation and test ids are rejected. The printed accuracy is computed on the
held-out official training pair only.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.ffpp_splits import assert_train_ids_only, load_frozen_regime, pair_ids
from src.models.xception import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    build_xception,
    freeze_backbone,
    unfreeze_all,
)
from src.pipeline.crop import CROP_FACTOR, TRACKER_NAME, crop_largest_face
from src.pipeline.decode import iter_bgr_frames

RELEASE_DIRS = {
    "original": "original_sequences/youtube",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
METHODS = (
    ("original", 0),
    ("Deepfakes", 1),
    ("Face2Face", 1),
    ("FaceSwap", 1),
    ("NeuralTextures", 1),
)
MAX_CROPS_PER_VIDEO = 16
FROZEN_EPOCHS = 1
FINETUNE_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.0002
PAPER_BATCH_SIZE = 32
ACCURACY_THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _files_for_pairs(data_root: Path, pairs: list[list[str]]) -> list[tuple[Path, str, int, str]]:
    rows = []
    for left, right in pairs:
        for video_id in (left, right):
            path = data_root / RELEASE_DIRS["original"] / "c23" / "videos" / f"{video_id}.mp4"
            rows.append((path, video_id, 0, "original"))
        for method, label in METHODS:
            if method == "original":
                continue
            for stem in (f"{left}_{right}", f"{right}_{left}"):
                path = data_root / RELEASE_DIRS[method] / "c23" / "videos" / f"{stem}.mp4"
                rows.append((path, stem, label, method))
    return rows


def _rgb_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        array[:, :, channel] = (array[:, :, channel] - mean) / std
    return torch.from_numpy(array).permute(2, 0, 1)


def collect_crops(data_root: Path, pairs: list[list[str]]) -> list[tuple[torch.Tensor, int]]:
    ids = pair_ids(pairs)
    assert_train_ids_only(ids)
    samples = []
    for path, _video_id, label, _method in _files_for_pairs(data_root, pairs):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing official c23 file {path}. This session does not invent crops or scores."
            )
        kept = 0
        for _index, frame in iter_bgr_frames(path):
            found = crop_largest_face(frame)
            if not found["ok"]:
                continue
            samples.append((_rgb_tensor(found["crop_image"]), label))
            kept += 1
            if kept >= MAX_CROPS_PER_VIDEO:
                break
        if kept == 0:
            print(f"No face crop in {path}. That video is left out of the tensors.")
    return samples


class CropSet(Dataset):
    def __init__(self, samples: list[tuple[torch.Tensor, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]
        return image, label


def _class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    counts = [labels.count(0), labels.count(1)]
    if counts[0] == 0 or counts[1] == 0:
        raise RuntimeError(f"Both classes need crops before training. Counts: real={counts[0]} fake={counts[1]}")
    total = float(sum(counts))
    weights = [total / (2.0 * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    fake_probability = torch.softmax(logits, dim=1)[:, 1]
    predicted = fake_probability >= ACCURACY_THRESHOLD
    return float((predicted == labels.bool()).float().mean().item())


def _run_epoch(model, loader, optimizer, weights, device) -> float:
    model.train()
    losses = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, weight=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Training epoch saw no batches.")
    return sum(losses) / len(losses)


def _evaluate(model, loader, device) -> float:
    model.eval()
    seen = 0
    score = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            count = int(labels.shape[0])
            seen += count
            score += _accuracy(logits, labels) * count
    if seen == 0:
        raise RuntimeError("Validation loader was empty. No accuracy will be printed.")
    return score / seen


def _fit_stage(model, train_loader, monitor_loader, stage: str, epochs: int, weights, device) -> list[dict]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, weights, device)
        val_accuracy = _evaluate(model, monitor_loader, device)
        row = {"stage": stage, "epoch": epoch, "train_loss": train_loss, "val_accuracy": val_accuracy}
        history.append(row)
        print(
            f"stage={stage} epoch={epoch} train_loss={train_loss:.6f} "
            f"val_accuracy={val_accuracy:.6f}"
        )
    return history


def train_regime_s(data_root: Path, checkpoint_path: Path, pretrained: bool = True, seed: int = 0) -> list[dict]:
    """Train on the frozen Regime S lists and return the computed history.

    ``val_accuracy`` is the 0.5-threshold accuracy on the held-out training
    pair. It is not the official 140-video validation set and not Table 1.
    """
    set_seed(seed)
    regime = load_frozen_regime()
    data_root = Path(data_root)
    train_samples = collect_crops(data_root, regime["train_pairs"])
    monitor_samples = collect_crops(data_root, regime["monitor_pairs"])
    if not train_samples or not monitor_samples:
        raise RuntimeError("Face crops were empty. Not printing a metric.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} tracker={TRACKER_NAME} crop_factor={CROP_FACTOR}")
    print(
        "Validation accuracy uses a 0.5 threshold on the fake-class softmax "
        "of the held-out official training pair. This is not Table 1."
    )
    model = build_xception(pretrained=pretrained).to(device)
    train_loader = DataLoader(CropSet(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    monitor_loader = DataLoader(CropSet(monitor_samples), batch_size=BATCH_SIZE, shuffle=False)
    weights = _class_weights([label for _image, label in train_samples], device)
    history = []
    best_accuracy = None
    best_state = None

    def _remember():
        nonlocal best_accuracy, best_state
        if not history:
            return
        latest = history[-1]["val_accuracy"]
        if best_accuracy is None or latest >= best_accuracy:
            best_accuracy = latest
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    freeze_backbone(model)
    history.extend(
        _fit_stage(model, train_loader, monitor_loader, "frozen_backbone", FROZEN_EPOCHS, weights, device)
    )
    _remember()
    unfreeze_all(model)
    history.extend(
        _fit_stage(model, train_loader, monitor_loader, "finetune", FINETUNE_EPOCHS, weights, device)
    )
    _remember()
    best = max(history, key=lambda row: row["val_accuracy"])
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "model_name": "legacy_xception",
            "pretrained_init": pretrained,
            "history": history,
            "best_monitor_accuracy": best["val_accuracy"],
            "tracker": TRACKER_NAME,
            "crop_factor": CROP_FACTOR,
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    print("The weight file stays out of git. A Colab ephemeral disk is not the backup.")
    return history
