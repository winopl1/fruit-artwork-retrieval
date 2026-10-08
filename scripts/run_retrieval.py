from __future__ import annotations

import argparse

from task3_mac.config import load_config
from task3_mac.data import read_records
from task3_mac.evaluate import evaluate_matches
from task3_mac.models import build_encoder
from task3_mac.retrieval import extract_embeddings, search_top_k, write_task3_output
from task3_mac.visualize import save_balanced_contact_sheet, save_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3_mac.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw

    label_candidates = list(raw["data"]["label_candidates"])
    extensions = list(raw["data"]["image_extensions"])
    image_subdir = str(raw["data"]["image_subdir"])

    query_records = read_records(cfg.data_root, cfg.query_split, image_subdir, label_candidates, extensions)
    if cfg.query_split == cfg.gallery_split:
        gallery_records = query_records
    else:
        gallery_records = read_records(cfg.data_root, cfg.gallery_split, image_subdir, label_candidates, extensions)

    if any(rec.style_label == "unknown" or rec.fruit_label == "unknown" for rec in gallery_records):
        raise ValueError("Gallery labels are required for Task3 output; check the label CSV")

    print("query split:", cfg.query_split, "images:", len(query_records))
    print("gallery split:", cfg.gallery_split, "images:", len(gallery_records))

    encoder = build_encoder(raw)
    print("encoder:", encoder.name)
    print("device:", encoder.device)

    batch_size = int(raw["model"]["batch_size"])
    num_workers = int(raw["model"]["num_workers"])
    query_embeddings = extract_embeddings(query_records, encoder, batch_size, num_workers)

    if cfg.query_split == cfg.gallery_split:
        gallery_embeddings = query_embeddings
    else:
        gallery_embeddings = extract_embeddings(gallery_records, encoder, batch_size, num_workers)

    indices = search_top_k(
        query_embeddings,
        gallery_embeddings,
        cfg.top_k,
        bool(raw["retrieval"]["exclude_self"]),
        query_records,
        gallery_records,
    )

    write_task3_output(cfg.output_path, indices, gallery_records)
    print("saved:", cfg.output_path)

    if bool(raw["evaluation"]["enabled"]):
        model_tag = str(raw["model"]["name"]).replace("/", "_")
        eval_path = cfg.output_dir / f"{model_tag}_{cfg.query_split}_retrieval_report.csv"
        evaluate_matches(query_records, gallery_records, indices, eval_path)
        print("saved:", eval_path)

    viz = raw.get("visualization", {})
    if bool(viz.get("enabled", False)):
        viz_path = cfg.output_dir / str(viz.get("output_name", "retrieval_contact_sheet.jpg"))
        save_contact_sheet(
            viz_path,
            query_records,
            gallery_records,
            indices,
            int(viz.get("max_queries", 12)),
            int(viz.get("tile_size", 128)),
        )
        print("saved:", viz_path)

        if bool(viz.get("balanced", False)):
            balanced_name = str(viz.get("balanced_output_name", "balanced_contact_sheet.jpg"))
            balanced_path = cfg.output_dir / balanced_name
            save_balanced_contact_sheet(
                balanced_path,
                query_records,
                gallery_records,
                indices,
                str(viz.get("group_field", "fruit_label")),
                int(viz.get("per_group", 2)),
                int(viz.get("max_queries", 12)),
                int(viz.get("tile_size", 128)),
            )
            print("saved:", balanced_path)

    with open(cfg.output_path, "r", encoding="utf-8") as f:
        for _, line in zip(range(3), f):
            print(line.strip())


if __name__ == "__main__":
    main()
