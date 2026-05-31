import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from src.utils.io import ensure_dir


def make_demo_image(index: int, total: int, size: int = 384) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    yy, xx = np.mgrid[0:size, 0:size]
    canvas[..., 0] = np.clip(40 + xx * 0.35, 0, 255)
    canvas[..., 1] = np.clip(45 + yy * 0.30, 0, 255)
    canvas[..., 2] = 95

    angle = 2.0 * np.pi * index / max(1, total)
    cx = int(size * 0.5 + np.cos(angle) * size * 0.12)
    cy = int(size * 0.5 + np.sin(angle) * size * 0.08)

    cv2.rectangle(canvas, (50, 250), (334, 310), (65, 75, 90), -1)
    cv2.circle(canvas, (cx, cy), 72, (218, 174, 72), -1)
    cv2.circle(canvas, (cx - 24, cy - 18), 18, (245, 230, 160), -1)
    cv2.rectangle(canvas, (cx - 55, cy + 58), (cx + 55, cy + 105), (120, 92, 62), -1)

    for k in range(7):
        x = 40 + k * 48 + int(np.sin(angle + k) * 8)
        cv2.line(canvas, (x, 40), (x + 36, 330), (235, 235, 220), 2)

    label = f"view {index:03d}"
    cv2.putText(canvas, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return canvas


def create_demo_scene(scene: str, images: int, output_root: str | Path) -> Path:
    image_dir = ensure_dir(Path(output_root) / scene / "images")
    for i in range(images):
        img = make_demo_image(i, images)
        Image.fromarray(img).save(image_dir / f"view_{i:03d}.png")
    return image_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="demo_scene")
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--output-root", default="datasets/raw")
    args = parser.parse_args()

    if args.images < 2:
        raise ValueError("Demo scene needs at least 2 images.")

    image_dir = create_demo_scene(args.scene, args.images, args.output_root)
    print(f"Saved {args.images} demo images to {image_dir}")


if __name__ == "__main__":
    main()
