from pathlib import Path
import random
from src.utils.io import ensure_dir, save_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    return images

def uniform_split(images: list[Path], n: int) -> list[str]:
    if n > len(images):
        raise ValueError(f"Requested {n} views but only {len(images)} images exist")
    if n == 1:
        return [images[len(images)//2].name]
    indices = [round(i * (len(images) - 1) / (n - 1)) for i in range(n)]
    return [images[i].name for i in indices]

def random_split(images: list[Path], n: int, seed: int = 42) -> list[str]:
    if n > len(images):
        raise ValueError(f"Requested {n} views but only {len(images)} images exist")
    rng = random.Random(seed)
    selected = sorted(rng.sample(images, n))
    return [p.name for p in selected]

def create_splits(scene_dir: str | Path, view_counts: list[int], output_dir: str | Path, seed: int = 42):
    scene_dir = Path(scene_dir)
    image_dir = scene_dir / "images"
    images = list_images(image_dir)
    output_dir = ensure_dir(output_dir)

    metadata = {
        "scene": scene_dir.name,
        "num_images": len(images),
        "splits": {}
    }

    for n in view_counts:
        metadata["splits"][f"{n}views_uniform"] = uniform_split(images, n)
        metadata["splits"][f"{n}views_random"] = random_split(images, n, seed)

    save_json(metadata, output_dir / f"{scene_dir.name}_splits.json")
    return metadata
