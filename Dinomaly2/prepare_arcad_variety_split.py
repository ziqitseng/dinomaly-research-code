#!/usr/bin/env python3
"""Create deterministic ArcAD-style cold-start manifests for Real-IAD Variety."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024"
)
DEFAULT_SOURCE_JSONS = Path("realiadvariety_jsons/Real-IAD_Variety_jsons")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "dataset" / "Real-IAD-Variety-ArcAD-Splits"
NORMAL_CLASS = "OK"
VIEWS_PER_OBJECT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Real-IAD Variety 1024 root containing category folders and source split JSONs.")
    parser.add_argument("--source-json-dir", type=Path, default=DEFAULT_SOURCE_JSONS,
                        help="Path under data-root containing official per-category train/test JSONs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Staging/output directory; manifests go in realiadvariety_jsons/sup/.")
    parser.add_argument("--normal-train-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def group_key(record: dict) -> str:
    # Variety stores the five camera views under a common defect-class/object folder.
    parts = Path(record["image_path"]).parts
    if len(parts) < 3:
        raise ValueError(f"Expected class/object/file path, got: {record['image_path']!r}")
    return "/".join(parts[:-1])


def grouped_records(records: list[dict], category: str, phase: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)
    for key, views in groups.items():
        if len(views) != VIEWS_PER_OBJECT:
            raise ValueError(f"{category}/{phase}/{key}: expected 5 views, found {len(views)}")
        classes = {view["anomaly_class"] for view in views}
        if len(classes) != 1:
            raise ValueError(f"{category}/{phase}/{key}: views have inconsistent anomaly classes {classes}")
    return dict(groups)


def category_image_root(data_root: Path, category: str) -> Path:
    image_root = data_root / category
    nested = image_root / category
    return nested if nested.is_dir() else image_root


def validate_records(records: list[dict], image_root: Path, category: str) -> None:
    for record in records:
        image_path = image_root / record["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing {category} image: {image_path}")
        mask_rel = record.get("mask_path")
        if mask_rel:
            mask_path = image_root / mask_rel
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing {category} mask: {mask_path}")


def allocate_normal_counts(categories: list[str], normal_groups: dict[str, dict[str, list[dict]]],
                           fraction: float, rng: random.Random) -> dict[str, int]:
    totals = {category: len(normal_groups[category]) for category in categories}
    target_total = int(math.floor(sum(totals.values()) * fraction + 0.5))
    exact = {category: totals[category] * fraction for category in categories}
    allocation = {category: int(math.floor(exact[category])) for category in categories}
    remainder = target_total - sum(allocation.values())
    tie_order = categories[:]
    rng.shuffle(tie_order)
    tie_rank = {category: index for index, category in enumerate(tie_order)}
    ranked = sorted(categories, key=lambda category: (-(exact[category] - allocation[category]), tie_rank[category]))
    for category in ranked[:remainder]:
        allocation[category] += 1
    return allocation


def arcad_record(record: dict) -> dict:
    """ArcAD's current loader infers labels from anomaly_class, so maskless views become OK."""
    source_class = record["anomaly_class"]
    label = int(source_class != NORMAL_CLASS and bool(record.get("mask_path")))
    result = dict(record)
    if source_class != NORMAL_CLASS and not label:
        result["source_anomaly_class"] = source_class
        result["anomaly_class"] = NORMAL_CLASS
    result["label"] = label
    return result


def main() -> None:
    args = parse_args()
    if not 0 < args.normal_train_fraction < 1:
        raise ValueError("normal-train-fraction must be in (0, 1).")
    data_root = args.data_root.expanduser().resolve()
    source_dir = (data_root / args.source_json_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source JSON directory not found: {source_dir}")
    source_files = sorted(source_dir.glob("*.json"))
    if not source_files:
        raise FileNotFoundError(f"No category JSON files in: {source_dir}")

    categories = [path.stem for path in source_files]
    source_data = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in source_files}
    train_normal_groups: dict[str, dict[str, list[dict]]] = {}
    test_normal_groups: dict[str, dict[str, list[dict]]] = {}
    test_anomaly_groups: dict[str, dict[str, list[dict]]] = {}

    for category in categories:
        splits = source_data[category]
        if not {"train", "test"}.issubset(splits):
            raise ValueError(f"{category}: official JSON needs train and test lists")
        train_records, test_records = splits["train"], splits["test"]
        unexpected_train = [r for r in train_records if r["anomaly_class"] != NORMAL_CLASS]
        if unexpected_train:
            raise ValueError(f"{category}: expected normal-only official train split")
        root = category_image_root(data_root, category)
        validate_records(train_records + test_records, root, category)
        train_groups = grouped_records(train_records, category, "train")
        test_groups = grouped_records(test_records, category, "test")
        train_normal_groups[category] = train_groups
        test_normal_groups[category] = {
            key: group for key, group in test_groups.items()
            if not any(record.get("mask_path") for record in group)
        }
        test_anomaly_groups[category] = {
            key: group for key, group in test_groups.items()
            if any(record.get("mask_path") for record in group)
        }
        if not test_normal_groups[category] or not test_anomaly_groups[category]:
            raise ValueError(f"{category}: test split needs both normal and anomalous objects")

    rng = random.Random(args.seed)
    normal_counts = allocate_normal_counts(categories, train_normal_groups, args.normal_train_fraction, rng)
    manifests_dir = args.output_dir.expanduser().resolve() / "realiadvariety_jsons" / "sup"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "Real-IAD Variety 1024",
        "protocol": "ArcAD-style cold start; 30% normal training objects plus one labeled defect object per category",
        "source_json_dir": str(source_dir),
        "seed": args.seed,
        "normal_train_fraction": args.normal_train_fraction,
        "views_per_object": VIEWS_PER_OBJECT,
        "category_count": len(categories),
        "categories": {},
    }
    total_labeled_normals = total_labeled_anomalies = total_test = 0
    total_train_normal_objects = total_test_anomaly_objects = 0

    for category in categories:
        train_groups = train_normal_groups[category]
        train_keys = sorted(train_groups)
        rng.shuffle(train_keys)
        selected_normal_keys = set(train_keys[:normal_counts[category]])
        remaining_normal_groups = [
            group for key, group in train_groups.items() if key not in selected_normal_keys
        ]

        anomaly_keys = sorted(test_anomaly_groups[category])
        selected_anomaly_key = rng.choice(anomaly_keys)
        selected_anomaly_group = test_anomaly_groups[category][selected_anomaly_key]

        labeled_records = [
            arcad_record(record) for key in sorted(selected_normal_keys)
            for record in train_groups[key]
        ] + [arcad_record(record) for record in selected_anomaly_group]
        test_records = [
            arcad_record(record) for group in remaining_normal_groups for record in group
        ]
        test_records += [
            arcad_record(record)
            for key, group in test_normal_groups[category].items() for record in group
        ]
        test_records += [
            arcad_record(record)
            for key, group in test_anomaly_groups[category].items()
            if key != selected_anomaly_key
            for record in group
        ]

        labeled_image_paths = {record["image_path"] for record in labeled_records}
        test_image_paths = {record["image_path"] for record in test_records}
        if labeled_image_paths & test_image_paths:
            raise RuntimeError(f"{category}: labeled/test image leakage detected")

        labeled_normal_images = sum(record["label"] == 0 for record in labeled_records)
        labeled_anomaly_images = sum(record["label"] == 1 for record in labeled_records)
        category_summary = {
            "official_train_normal_objects": len(train_groups),
            "labeled_normal_objects": len(selected_normal_keys),
            "labeled_anomaly_objects": 1,
            "labeled_normal_images": labeled_normal_images,
            "labeled_anomaly_images": labeled_anomaly_images,
            "test_images": len(test_records),
            "heldout_test_normal_images": sum(record["label"] == 0 for record in test_records),
            "heldout_test_anomaly_images": sum(record["label"] == 1 for record in test_records),
            "labeled_anomaly_fraction": labeled_anomaly_images / len(labeled_records),
            "selected_labeled_anomaly_object": selected_anomaly_key,
        }
        summary["categories"][category] = category_summary
        total_labeled_normals += labeled_normal_images
        total_labeled_anomalies += labeled_anomaly_images
        total_test += len(test_records)
        total_train_normal_objects += len(train_groups)
        total_test_anomaly_objects += len(test_anomaly_groups[category])

        manifest = {
            "meta": {
                "dataset": "realiad-variety",
                "category": category,
                "seed": args.seed,
                "protocol": "arcad-cold-start-variety-v1",
                "normal_train_fraction": args.normal_train_fraction,
                "views_per_object": VIEWS_PER_OBJECT,
                "num_labeled": len(labeled_records),
                "num_test": len(test_records),
                "num_labeled_normal_objects": len(selected_normal_keys),
                "num_labeled_anomaly_objects": 1,
            },
            "labeled": labeled_records,
            "test": test_records,
        }
        (manifests_dir / f"{category}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    summary["totals"] = {
        "normal_train_objects": total_train_normal_objects,
        "labeled_normal_images": total_labeled_normals,
        "labeled_anomaly_images": total_labeled_anomalies,
        "labeled_anomaly_fraction": total_labeled_anomalies / (total_labeled_normals + total_labeled_anomalies),
        "labeled_anomaly_objects": len(categories),
        "test_images": total_test,
        "original_test_anomaly_objects": total_test_anomaly_objects,
    }
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    (args.output_dir.expanduser().resolve() / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir.expanduser().resolve() / "README.md").write_text(
        "# Real-IAD Variety ArcAD-style cold-start split\n\n"
        "Generated from the official Variety per-category train/test manifests. "
        "Per category, approximately 30% of normal training object groups are selected as labeled normals; "
        "one anomalous object group is selected from the official test split as labeled defect supervision. "
        "All five camera views of each object stay in one partition. Remaining official test objects plus "
        "the unselected 70% of normal training objects form the held-out test set.\n\n"
        "The one-defect-object-per-category rule preserves defect supervision for all 160 classes. "
        "Because of the five-view grouping and the number of categories, this yields an overall labeled "
        "anomaly fraction near the paper's nominal 10% target; see `summary.json` for exact counts. "
        "Views whose source `mask_path` is null are encoded as normal views, matching Dinomaly2's Variety "
        "test loader behavior while retaining the original class in `source_anomaly_class`.\n\n"
        "Manifest records preserve the source `image_path`, `mask_path`, and `anomaly_class` fields, "
        "and add integer `label` values (0 normal, 1 anomaly). The JSON keys are `labeled` and `test`.\n\n"
        "These manifests are stored separately from the official Variety train/test manifests. "
        "ArcAD's current Real-IAD loader assumes the standard `realiad_1024` image layout; a Variety-specific "
        "loader/path adjustment is still needed before using these files directly.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(categories)} category manifests to: {manifests_dir}")
    print(f"Summary: {args.output_dir.expanduser().resolve() / 'summary.json'}")
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
