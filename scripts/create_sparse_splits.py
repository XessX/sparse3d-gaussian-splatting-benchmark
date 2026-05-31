import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import create_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Scene name inside datasets/raw")
    parser.add_argument("--views", nargs="+", type=int, default=[2, 3, 5, 8, 12, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scene_dir = Path("datasets/raw") / args.scene
    output_dir = Path("datasets/processed") / args.scene / "splits"
    metadata = create_splits(scene_dir, args.views, output_dir, args.seed)
    print(f"Saved splits for {metadata['scene']} with {metadata['num_images']} images.")

if __name__ == "__main__":
    main()
