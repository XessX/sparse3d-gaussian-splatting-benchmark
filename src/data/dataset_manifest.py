from pathlib import Path
import json
from src.data.splits import list_images
from src.utils.io import ensure_dir

def create_scene_manifest(scene_dir: str | Path, output_path: str | Path):
    scene_dir = Path(scene_dir)
    images = list_images(scene_dir / "images")
    manifest = {
        "scene_name": scene_dir.name,
        "image_dir": str(scene_dir / "images"),
        "num_images": len(images),
        "images": [p.name for p in images],
        "notes": "Add camera metadata, test images, and ground-truth paths when available."
    }
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
