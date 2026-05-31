from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from src.utils.io import ensure_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def iter_images(image_dir: str | Path):
    image_dir = Path(image_dir)
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

def apply_motion_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def adjust_exposure(img: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def save_jpeg_with_quality(img_rgb: np.ndarray, output_path: str | Path, quality: int):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    Image.fromarray(img_rgb).save(output_path, quality=quality)

def degrade_scene(
    input_image_dir: str | Path,
    output_root: str | Path,
    resize_scales=(0.25, 0.50, 0.75),
    jpeg_quality=(20, 40, 60, 80),
    motion_blur_kernel=(7, 15),
    exposure_alpha=(0.6, 1.4),
    gaussian_noise_sigma=(5, 15),
    seed: int = 42,
    max_images: int | None = None,
):
    input_image_dir = Path(input_image_dir)
    output_root = Path(output_root)
    rng = np.random.default_rng(seed)
    images = iter_images(input_image_dir)
    if not images:
        raise ValueError(f"No images found in {input_image_dir}")
    if max_images is not None:
        images = images[:max_images]

    for image_path in images:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for scale in resize_scales:
            out = resize_image(rgb, scale)
            out_dir = ensure_dir(output_root / f"resize_{int(scale*100)}")
            Image.fromarray(out).save(out_dir / image_path.name)

        for q in jpeg_quality:
            out_dir = ensure_dir(output_root / f"jpeg_q{q}")
            save_jpeg_with_quality(rgb, out_dir / f"{image_path.stem}.jpg", q)

        for k in motion_blur_kernel:
            out = apply_motion_blur(rgb, k)
            out_dir = ensure_dir(output_root / f"motion_blur_k{k}")
            Image.fromarray(out).save(out_dir / image_path.name)

        for alpha in exposure_alpha:
            out = adjust_exposure(rgb, alpha)
            tag = "under" if alpha < 1 else "over"
            out_dir = ensure_dir(output_root / f"exposure_{tag}_{alpha}")
            Image.fromarray(out).save(out_dir / image_path.name)

        for sigma in gaussian_noise_sigma:
            out = add_gaussian_noise(rgb, sigma, rng)
            out_dir = ensure_dir(output_root / f"noise_sigma{sigma}")
            Image.fromarray(out).save(out_dir / image_path.name)

    return {
        "input_image_dir": str(input_image_dir),
        "output_root": str(output_root),
        "num_images": len(images),
        "filenames": [p.name for p in images],
        "conditions": {
            "resize_scales": list(resize_scales),
            "jpeg_quality": list(jpeg_quality),
            "motion_blur_kernel": list(motion_blur_kernel),
            "exposure_alpha": list(exposure_alpha),
            "gaussian_noise_sigma": list(gaussian_noise_sigma),
        },
        "seed": seed,
    }
