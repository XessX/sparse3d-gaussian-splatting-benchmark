import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


SCENES = [
    {
        "dataset": "Tanks and Temples",
        "scene": "tandt/train",
        "source": "Official GraphDeco 3DGS public dataset package / Tanks and Temples scene",
        "license_notes": "Use dataset-specific terms and cite Tanks and Temples; verify license page before public redistribution.",
        "dataset_status": "public dataset",
        "redistribution_status": "MANUAL CHECK REQUIRED",
        "raw_images_included": "not included in lightweight package",
        "citation_required": "yes",
    },
    {
        "dataset": "Tanks and Temples",
        "scene": "tandt/truck",
        "source": "Official GraphDeco 3DGS public dataset package / Tanks and Temples scene",
        "license_notes": "Use dataset-specific terms and cite Tanks and Temples; verify license page before public redistribution.",
        "dataset_status": "public dataset",
        "redistribution_status": "MANUAL CHECK REQUIRED",
        "raw_images_included": "not included in lightweight package",
        "citation_required": "yes",
    },
    {
        "dataset": "Deep Blending",
        "scene": "db/drjohnson",
        "source": "Official GraphDeco 3DGS public dataset package / Deep Blending scene",
        "license_notes": "Use dataset-specific terms and cite Deep Blending/3DGS dataset package; verify redistribution terms before release.",
        "dataset_status": "public dataset",
        "redistribution_status": "MANUAL CHECK REQUIRED",
        "raw_images_included": "not included in lightweight package",
        "citation_required": "yes",
    },
    {
        "dataset": "Deep Blending",
        "scene": "db/playroom",
        "source": "Official GraphDeco 3DGS public dataset package / Deep Blending scene",
        "license_notes": "Use dataset-specific terms and cite Deep Blending/3DGS dataset package; verify redistribution terms before release.",
        "dataset_status": "public dataset",
        "redistribution_status": "MANUAL CHECK REQUIRED",
        "raw_images_included": "not included in lightweight package",
        "citation_required": "yes",
    },
]


def scene_leaf(scene: str) -> str:
    return Path(scene).name


def count_images(scene: str) -> int:
    image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
    return len([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def eval_test_count(scene: str) -> int:
    path = PROJECT_ROOT / "datasets" / "processed" / scene / "eval_test_views.json"
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")).get("test_images", []))


def split_counts(scene: str) -> tuple[int, str]:
    path = PROJECT_ROOT / "datasets" / "processed" / scene / "splits" / f"{scene_leaf(scene)}_splits.json"
    if not path.exists():
        return 0, ""
    data = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(data.get("splits", {}))
    selection_types = sorted({name.split("views_", 1)[1] for name in names if "views_" in name})
    return len(names), ", ".join(selection_types)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "scene",
        "image_count",
        "held_out_test_views",
        "split_count",
        "dataset_status",
        "redistribution_status",
        "raw_images_included",
        "citation_required",
        "license_notes",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in df[columns].iterrows()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    rows = []
    for item in SCENES:
        split_count, selection_types = split_counts(item["scene"])
        rows.append(
            {
                **item,
                "image_count": count_images(item["scene"]),
                "held_out_test_views": eval_test_count(item["scene"]),
                "split_count": split_count,
                "selection_types": selection_types,
            }
        )
    df = pd.DataFrame(rows)
    table_dir = ensure_dir(PROJECT_ROOT / "manuscript" / "tables")
    csv_path = table_dir / "table_dataset_summary.csv"
    md_path = table_dir / "table_dataset_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Table: Dataset summary\n\n"
        + markdown_table(df)
        + "\n\n"
        "Note: Mip-NeRF 360/LLFF/DTU are configured as candidate extensions but were not downloaded in this phase because of dataset size and setup cost.\n",
        encoding="utf-8",
    )
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
