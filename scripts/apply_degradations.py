import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation.image_degradations import degrade_scene
from src.utils.io import save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--input-subdir", default="images", help="Usually images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None, help="Limit processing for smoke tests")
    args = parser.parse_args()

    input_dir = Path("datasets/raw") / args.scene / args.input_subdir
    output_root = Path("datasets/processed") / args.scene / "degraded"
    metadata = degrade_scene(input_dir, output_root, seed=args.seed, max_images=args.max_images)
    save_json(metadata, output_root / "degradation_metadata.json")
    print(f"Saved degraded images to {output_root}")
    print(f"Processed {metadata['num_images']} images.")

if __name__ == "__main__":
    main()
