import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import list_images
from src.utils.io import ensure_dir, save_json


def uniform_select(items: list[str], n: int) -> list[str]:
    if n >= len(items):
        return items
    if n == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[i] for i in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", default=None)
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
    avoid = {name for split in split_data.get("splits", {}).values() for name in split}
    all_images = [p.name for p in list_images(image_dir)]
    candidates = [name for name in all_images if name not in avoid]
    selected = uniform_select(candidates, args.count)

    output = (
        PROJECT_ROOT / args.output
        if args.output
        else PROJECT_ROOT / "datasets" / "processed" / args.scene / "eval_test_views.json"
    )
    ensure_dir(output.parent)
    save_json(
        {
            "scene": args.scene,
            "count": len(selected),
            "avoid_train_split_images": len(avoid),
            "candidate_count": len(candidates),
            "test_images": selected,
        },
        output,
    )
    print(f"Saved {len(selected)} held-out test views to {output}")


if __name__ == "__main__":
    main()
