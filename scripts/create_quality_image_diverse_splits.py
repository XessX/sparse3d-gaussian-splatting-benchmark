import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selection.image_diversity import select_quality_image_diverse
from src.utils.io import ensure_dir, save_csv, save_json


def scene_leaf(scene: str) -> str:
    return Path(scene).name


def load_eval_test_images(scene: str) -> set[str]:
    path = PROJECT_ROOT / "datasets" / "processed" / scene / "eval_test_views.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("test_images", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["tandt/train", "tandt/truck", "db/drjohnson", "db/playroom"])
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 5, 8, 12, 20])
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--quality-weight", type=float, default=0.45)
    parser.add_argument("--diversity-weight", type=float, default=0.55)
    args = parser.parse_args()

    summary_rows = []
    for scene in args.scenes:
        leaf = scene_leaf(scene)
        image_dir = PROJECT_ROOT / "datasets" / "raw" / scene / "images"
        split_path = PROJECT_ROOT / "datasets" / "processed" / scene / "splits" / f"{leaf}_splits.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        excluded = load_eval_test_images(scene)
        scene_rows = []
        for n in args.views:
            selected, rows = select_quality_image_diverse(
                image_dir,
                n,
                exclude=excluded,
                candidate_pool=args.candidate_pool,
                quality_weight=args.quality_weight,
                diversity_weight=args.diversity_weight,
            )
            split_name = f"{n}views_quality_image_diverse"
            split_data["splits"][split_name] = selected
            for row in rows:
                scene_rows.append(
                    {
                        "scene": scene,
                        "split": split_name,
                        "views": n,
                        **row,
                    }
                )
            summary_rows.append(
                {
                    "scene": scene,
                    "split": split_name,
                    "views": n,
                    "selected_images": json.dumps(selected),
                    "excluded_test_images": len(excluded),
                    "candidate_pool": args.candidate_pool,
                    "quality_weight": args.quality_weight,
                    "diversity_weight": args.diversity_weight,
                }
            )

        split_data["quality_image_diverse_selection"] = {
            "quality_weight": args.quality_weight,
            "diversity_weight": args.diversity_weight,
            "candidate_pool": args.candidate_pool,
            "excluded_eval_test_views": len(excluded),
            "notes": "Pose-free greedy selection using image quality and image-space diversity from color histogram and ORB descriptor distances. Full-scene COLMAP camera positions are not used.",
        }
        split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")

        out_dir = ensure_dir(PROJECT_ROOT / "results" / "csv" / scene)
        save_csv(scene_rows, out_dir / f"{leaf}_quality_image_diverse_scores.csv")
        print(f"Updated {split_path}")

    summary_path = PROJECT_ROOT / "results" / "csv" / "quality_image_diverse_splits.csv"
    save_csv(summary_rows, summary_path)
    save_json(
        {
            "split_type": "quality_image_diverse",
            "scenes": args.scenes,
            "views": args.views,
            "quality_weight": args.quality_weight,
            "diversity_weight": args.diversity_weight,
            "candidate_pool": args.candidate_pool,
            "notes": "The selector is image-only and excludes existing held-out eval views when present.",
        },
        PROJECT_ROOT / "results" / "csv" / "quality_image_diverse_splits_metadata.json",
    )
    print(f"Saved image-diverse split summary to {summary_path}")


if __name__ == "__main__":
    main()
