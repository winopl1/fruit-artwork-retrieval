from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


@dataclass
class Encoder:
    name: str
    model: nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    device: torch.device


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet_encoder(config: dict) -> Encoder:
    device = pick_device(str(config["model"].get("device", "auto")))
    weights_name = str(config["model"].get("resnet_weights", "DEFAULT"))
    weights = models.ResNet18_Weights.DEFAULT if weights_name == "DEFAULT" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms() if weights else transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return Encoder("resnet18", model, preprocess, device)


def build_open_clip_encoder(config: dict) -> Encoder:
    import open_clip

    device = pick_device(str(config["model"].get("device", "auto")))
    model_name = str(config["model"]["open_clip_model"])
    pretrained = str(config["model"]["open_clip_pretrained"])
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return Encoder(f"open_clip:{model_name}", model, preprocess, device)


def build_encoder(config: dict) -> Encoder:
    name = str(config["model"]["name"]).lower()
    if name == "open_clip":
        return build_open_clip_encoder(config)
    if name == "resnet18":
        return build_resnet_encoder(config)
    raise ValueError(f"Unsupported model name: {name}")


@torch.no_grad()
def encode_batch(encoder: Encoder, images: torch.Tensor) -> torch.Tensor:
    images = images.to(encoder.device)
    if encoder.name.startswith("open_clip"):
        features = encoder.model.encode_image(images)
    else:
        features = encoder.model(images)
    return F.normalize(features.float(), p=2, dim=1).cpu()
