from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: Path
    style_label: str = "unknown"
    fruit_label: str = "unknown"


def natural_key(path: str | Path) -> list[object]:
    stem = Path(path).stem
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", stem)]


def clean_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def image_id(path_or_name: str | Path) -> str:
    return clean_value(Path(clean_value(path_or_name)).stem)


def is_image(path: Path, extensions: Iterable[str]) -> bool:
    if path.suffix.lower() in set(extensions):
        return True
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def split_image_dir(data_root: Path, split: str, image_subdir: str) -> Path:
    nested = data_root / split / image_subdir
    if nested.exists():
        return nested
    return data_root / split


def find_label_csv(data_root: Path, split: str, candidates: Iterable[str]) -> Path | None:
    split_dir = data_root / split
    for name in candidates:
        path = split_dir / name
        if path.exists():
            return path
    return None


def pick_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def read_records(data_root: Path, split: str, image_subdir: str, label_candidates: list[str], extensions: list[str]) -> list[ImageRecord]:
    image_dir = split_image_dir(data_root, split, image_subdir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    image_paths = sorted(
        [p for p in image_dir.rglob("*") if p.is_file() and is_image(p, extensions)],
        key=natural_key,
    )
    image_by_name = {p.name: p for p in image_paths}
    image_by_stem = {p.stem: p for p in image_paths}

    label_path = find_label_csv(data_root, split, label_candidates)
    if label_path is None:
        return [ImageRecord(id=image_id(p.name), path=p) for p in image_paths]

    df = pd.read_csv(label_path)
    file_col = pick_column(df, ["file_name", "filename", "image", "image_name", "img", "path", "id"])
    style_col = pick_column(df, ["style_label", "style", "style_id", "label_style"])
    fruit_col = pick_column(df, ["fruit_label", "fruit", "fruit_id", "label_fruit"])

    if file_col is None:
        return [ImageRecord(id=image_id(p.name), path=p) for p in image_paths]

    records: list[ImageRecord] = []
    for _, row in df.iterrows():
        file_name = clean_value(row[file_col])
        key = Path(file_name).name
        stem = Path(file_name).stem
        path = image_by_name.get(key) or image_by_stem.get(stem)
        if path is None:
            for ext in extensions:
                path = image_by_name.get(f"{file_name}{ext}")
                if path is not None:
                    break
        if path is None:
            raise ValueError(f"Label row has no matching image: {file_name}")
        records.append(
            ImageRecord(
                id=image_id(file_name),
                path=path,
                style_label=clean_value(row[style_col]) if style_col else "unknown",
                fruit_label=clean_value(row[fruit_col]) if fruit_col else "unknown",
            )
        )
    if len(records) != len(image_paths) or len({rec.path for rec in records}) != len(image_paths):
        raise ValueError(f"Labels in {label_path} must cover each image exactly once")
    return sorted(records, key=lambda rec: natural_key(rec.path))
