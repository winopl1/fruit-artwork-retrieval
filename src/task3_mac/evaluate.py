from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import ImageRecord


def evaluate_matches(
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    indices: np.ndarray,
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for query, row in zip(query_records, indices):
        retrieved = [gallery_records[int(i)] for i in row]
        fruit_hits = [r.fruit_label == query.fruit_label for r in retrieved]
        style_hits = [r.style_label == query.style_label for r in retrieved]
        both_hits = [a and b for a, b in zip(fruit_hits, style_hits)]
        rows.append(
            {
                "query_id": query.id,
                "fruit_label": query.fruit_label,
                "style_label": query.style_label,
                "precision_fruit": sum(fruit_hits) / len(fruit_hits),
                "precision_style": sum(style_hits) / len(style_hits),
                "precision_both": sum(both_hits) / len(both_hits),
                "top_ids": " ".join(r.id for r in retrieved),
            }
        )

    if not rows:
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
