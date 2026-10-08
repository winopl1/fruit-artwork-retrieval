from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import ImageRecord


def _load_thumb(path: Path, tile_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((tile_size, tile_size))
    canvas = Image.new("RGB", (tile_size, tile_size), "white")
    x = (tile_size - image.width) // 2
    y = (tile_size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def save_contact_sheet(
    output_path: Path,
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    indices: np.ndarray,
    max_queries: int,
    tile_size: int,
) -> None:
    rows = min(max_queries, len(query_records))
    cols = 1 + indices.shape[1]
    label_h = 24
    cell_w = tile_size
    cell_h = tile_size + label_h

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_idx in range(rows):
        query = query_records[row_idx]
        row_records = [query] + [gallery_records[int(i)] for i in indices[row_idx]]
        for col_idx, rec in enumerate(row_records):
            x = col_idx * cell_w
            y = row_idx * cell_h
            thumb = _load_thumb(rec.path, tile_size)
            sheet.paste(thumb, (x, y))
            prefix = "Q" if col_idx == 0 else f"T{col_idx}"
            label = f"{prefix} {rec.id} s{rec.style_label} f{rec.fruit_label}"
            draw.text((x + 4, y + tile_size + 5), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def balanced_query_indices(
    query_records: list[ImageRecord],
    group_field: str,
    per_group: int,
    max_queries: int,
) -> list[int]:
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(query_records):
        value = getattr(record, group_field)
        groups.setdefault(value, []).append(idx)

    selected: list[int] = []
    for value in sorted(groups, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        selected.extend(groups[value][:per_group])
        if len(selected) >= max_queries:
            return selected[:max_queries]
    return selected


def save_balanced_contact_sheet(
    output_path: Path,
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    indices: np.ndarray,
    group_field: str,
    per_group: int,
    max_queries: int,
    tile_size: int,
) -> None:
    selected = balanced_query_indices(query_records, group_field, per_group, max_queries)
    if not selected:
        return

    rows = len(selected)
    cols = 1 + indices.shape[1]
    label_h = 24
    cell_w = tile_size
    cell_h = tile_size + label_h

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_idx, query_idx in enumerate(selected):
        query = query_records[query_idx]
        row_records = [query] + [gallery_records[int(i)] for i in indices[query_idx]]
        for col_idx, rec in enumerate(row_records):
            x = col_idx * cell_w
            y = row_idx * cell_h
            thumb = _load_thumb(rec.path, tile_size)
            sheet.paste(thumb, (x, y))
            prefix = "Q" if col_idx == 0 else f"T{col_idx}"
            label = f"{prefix} {rec.id} s{rec.style_label} f{rec.fruit_label}"
            draw.text((x + 4, y + tile_size + 5), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
