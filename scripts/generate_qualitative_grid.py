import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def image_path(scene: str, split: str, group: str, image_name: str) -> Path:
    base = PROJECT_ROOT / "results" / "method_outputs" / scene / "3dgs_sparse" / split / "test" / "ours_3000"
    return base / group / image_name


def make_grid(scene: str, output_path: Path, image_names: list[str], tile_size: tuple[int, int]) -> None:
    rows = [
        ("Ground truth", "5views_uniform", "gt"),
        ("Uniform", "5views_uniform", "renders"),
        ("Random", "5views_random", "renders"),
        ("Quality-aware", "5views_quality", "renders"),
        ("Quality+diverse", "5views_quality_diverse", "renders"),
    ]
    label_w = 170
    header_h = 54
    tile_w, tile_h = tile_size
    width = label_w + tile_w * len(image_names)
    height = header_h + tile_h * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    small_font = load_font(16)

    for col, name in enumerate(image_names):
        x = label_w + col * tile_w
        draw.text((x + 12, 16), name, fill=(30, 30, 30), font=small_font)

    for row_idx, (label, split, group) in enumerate(rows):
        y = header_h + row_idx * tile_h
        draw.rectangle((0, y, width, y + tile_h), outline=(230, 230, 230))
        draw.text((14, y + tile_h // 2 - 12), label, fill=(20, 20, 20), font=font)
        for col, name in enumerate(image_names):
            x = label_w + col * tile_w
            path = image_path(scene, split, group, name)
            if path.exists():
                tile = resize_crop(Image.open(path), tile_size)
            else:
                tile = Image.new("RGB", tile_size, (245, 245, 245))
                ImageDraw.Draw(tile).text((20, 20), "missing", fill=(160, 0, 0), font=font)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(255, 255, 255), width=2)

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="tandt/train")
    parser.add_argument("--images", nargs="+", default=["00000.png", "00005.png", "00010.png", "00015.png"])
    parser.add_argument("--tile-width", type=int, default=260)
    parser.add_argument("--tile-height", type=int, default=170)
    parser.add_argument("--output", default="results/qualitative/3dgs_eval_5views_grid.png")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    make_grid(args.scene, output_path, args.images, (args.tile_width, args.tile_height))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
