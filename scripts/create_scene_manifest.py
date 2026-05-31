import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manifest import create_scene_manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    scene_dir = Path("datasets/raw") / args.scene
    out_path = Path("datasets/processed") / args.scene / "manifest.json"
    manifest = create_scene_manifest(scene_dir, out_path)
    print(f"Saved manifest for {manifest['scene_name']} with {manifest['num_images']} images.")

if __name__ == "__main__":
    main()
