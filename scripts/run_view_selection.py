import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selection.quality_scores import rank_images_by_quality, select_top_quality_views
from src.utils.io import save_csv, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", type=int, default=5)
    args = parser.parse_args()

    image_dir = Path("datasets/raw") / args.scene / "images"
    ranked = rank_images_by_quality(image_dir)
    selected = select_top_quality_views(image_dir, args.views)

    out_dir = Path("results/csv")
    save_csv(ranked, out_dir / f"{args.scene}_quality_scores.csv")
    save_json({"scene": args.scene, "views": args.views, "selected": selected}, out_dir / f"{args.scene}_{args.views}views_quality_selection.json")
    print(f"Selected top {args.views} quality-aware views:")
    for s in selected:
        print(" -", s)

if __name__ == "__main__":
    main()
