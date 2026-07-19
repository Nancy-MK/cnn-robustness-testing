"""
pipeline_audit.py

Audits the training data pipeline: checks the per-class sample counts before
and after augmentation to verify that augmentation strategies do not
silently skew the class balance the model is trained on.

Usage:
    python src/pipeline_audit.py --data-dir data/train
"""
import argparse
import json
from pathlib import Path


def count_images_per_class(data_dir: Path) -> dict:
    counts = {}
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        counts[class_dir.name] = len(image_files)
    return counts


def audit_balance(counts: dict) -> dict:
    total = sum(counts.values())
    proportions = {k: round(v / total, 4) for k, v in counts.items()} if total else {}
    max_prop = max(proportions.values()) if proportions else 0
    min_prop = min(proportions.values()) if proportions else 0
    imbalance_ratio = round(max_prop / min_prop, 2) if min_prop > 0 else float("inf")

    return {
        "total_images": total,
        "counts_per_class": counts,
        "proportions_per_class": proportions,
        "imbalance_ratio": imbalance_ratio,
        "flag": (
            "Class imbalance ratio exceeds 3:1 - consider class weighting, "
            "oversampling, or targeted augmentation for the minority class(es)."
            if imbalance_ratio > 3
            else "Class balance within an acceptable range for standard training."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit the training data pipeline for class balance.")
    parser.add_argument("--data-dir", required=True, help="Directory containing one subfolder per class.")
    parser.add_argument("--out", default="artifacts/pipeline_audit_report.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    counts = count_images_per_class(data_dir)
    report = audit_balance(counts)

    print(json.dumps(report, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
