from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.copy()


def center_crop_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def standardize_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> Image.Image:
    rgb = image.convert("RGB")
    square = center_crop_to_square(rgb)
    return square.resize(size, Image.Resampling.LANCZOS)


def save_png(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG", optimize=False)


def standardize_and_save(
    input_path: Path,
    output_path: Path,
    size: tuple[int, int] = (512, 512),
) -> None:
    image = load_image(input_path)
    standardized = standardize_image(image, size=size)
    save_png(standardized, output_path)


def list_pngs(path: Path) -> Iterable[Path]:
    return sorted(path.glob("*.png"))
