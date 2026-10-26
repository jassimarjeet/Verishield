"""ImageNet XceptionNet with a two-class head.

torchvision 0.29 has no Xception class. This uses timm's ``legacy_xception``,
which is Chollet's network at 299x299. ``pretrained=False`` skips the weight
download and is only for the local smoke test.
"""

import timm
import torch

INPUT_SIZE = 299
MODEL_NAME = "legacy_xception"
IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def build_xception(pretrained: bool) -> torch.nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=2)


def freeze_backbone(model: torch.nn.Module) -> None:
    classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in classifier_ids


def unfreeze_all(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
