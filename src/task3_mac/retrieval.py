from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import ImageRecord
from .models import Encoder, encode_batch


class RecordDataset(Dataset):
    def __init__(self, records: list[ImageRecord], encoder: Encoder):
        self.records = records
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.records[idx].path).convert("RGB")
        return self.encoder.preprocess(image)


def extract_embeddings(records: list[ImageRecord], encoder: Encoder, batch_size: int, num_workers: int) -> torch.Tensor:
    loader = DataLoader(
        RecordDataset(records, encoder),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    chunks = []
    for images in tqdm(loader, desc=f"Encoding {len(records)} images"):
        chunks.append(encode_batch(encoder, images))
    if not chunks:
        raise ValueError("No images to encode.")
    return torch.cat(chunks, dim=0)


def search_top_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    top_k: int,
    exclude_self: bool,
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
) -> np.ndarray:
    similarity = query_embeddings @ gallery_embeddings.T

    if exclude_self:
        gallery_pos = {(rec.path.resolve(), rec.id): i for i, rec in enumerate(gallery_records)}
        for q_idx, q_rec in enumerate(query_records):
            g_idx = gallery_pos.get((q_rec.path.resolve(), q_rec.id))
            if g_idx is not None:
                similarity[q_idx, g_idx] = -float("inf")

    k = min(top_k, gallery_embeddings.shape[0])
    if exclude_self and query_records is gallery_records:
        k = min(top_k, max(0, gallery_embeddings.shape[0] - 1))
    if k <= 0:
        raise ValueError("Top-K is zero after excluding self.")
    return torch.topk(similarity, k=k, dim=1).indices.numpy()


def write_task3_output(path: Path, indices: np.ndarray, gallery_records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in indices:
            items = []
            for idx in row:
                rec = gallery_records[int(idx)]
                items.append(f"({rec.id},{rec.style_label},{rec.fruit_label})")
            f.write("[" + ", ".join(items) + "]\n")
