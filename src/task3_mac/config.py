from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Task3Config:
    raw: dict[str, Any]
    root: Path

    @property
    def student_id(self) -> str:
        return str(self.raw.get("student_id", "202301774"))

    @property
    def data_root(self) -> Path:
        return self.root / self.raw["paths"]["data_root"]

    @property
    def output_dir(self) -> Path:
        return self.root / self.raw["paths"]["output_dir"]

    @property
    def work_dir(self) -> Path:
        return self.root / self.raw["paths"]["work_dir"]

    @property
    def query_split(self) -> str:
        return str(self.raw["paths"]["query_split"])

    @property
    def gallery_split(self) -> str:
        return str(self.raw["paths"]["gallery_split"])

    @property
    def top_k(self) -> int:
        return int(self.raw["retrieval"]["top_k"])

    @property
    def output_path(self) -> Path:
        name = self.raw["retrieval"]["output_name"].format(student_id=self.student_id)
        return self.output_dir / name


def load_config(path: str | Path) -> Task3Config:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Task3Config(raw=raw, root=config_path.parents[1])
