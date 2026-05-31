import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selection.quality_scores import rank_images_by_quality
from src.utils.io import save_csv, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 5, 8, 12, 20])
    args = parser.parse_args()

    scene_leaf = Path(args.scene).name
    image_dir = PROJECT_ROOT / "datasets" / "raw" / args.scene / "images"
    split_path = (
        PROJECT_ROOT
        / "datasets"
        / "processed"
        / args.scene
        / "splits"
        / f"{scene_leaf}_splits.json"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    ranked = rank_images_by_quality(image_dir)

    for n in args.views:
        if n > len(ranked):
            raise ValueError(f"Requested {n} views but only {len(ranked)} images exist.")
        split_data["splits"][f"{n}views_quality"] = [row["filename"] for row in ranked[:n]]

    split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")

    out_dir = PROJECT_ROOT / "results" / "csv" / args.scene
    save_csv(ranked, out_dir / f"{scene_leaf}_quality_scores.csv")
    save_json(
        {
            "scene": args.scene,
            "views": args.views,
            "split_file": str(split_path.relative_to(PROJECT_ROOT)),
            "quality_split_keys": [f"{n}views_quality" for n in args.views],
        },
        out_dir / f"{scene_leaf}_quality_splits.json",
    )
    print(f"Updated {split_path}")
    print(f"Added quality splits: {', '.join(f'{n}views_quality' for n in args.views)}")


if __name__ == "__main__":
    main()
