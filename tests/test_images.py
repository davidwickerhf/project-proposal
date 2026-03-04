from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.images import (
    center_crop_to_square,
    list_pngs,
    load_image,
    save_png,
    standardize_and_save,
    standardize_image,
)


def test_center_crop_to_square_uses_min_side() -> None:
    image = Image.new("RGB", (13, 9), color=(255, 0, 0))
    cropped = center_crop_to_square(image)
    assert cropped.size == (9, 9)


def test_standardize_image_converts_to_rgb_and_target_size() -> None:
    image = Image.new("L", (100, 40), color=127)
    standardized = standardize_image(image, size=(64, 64))
    assert standardized.mode == "RGB"
    assert standardized.size == (64, 64)


def test_standardize_and_save_writes_png(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jpg"
    output_path = tmp_path / "out" / "final.png"

    Image.new("RGB", (32, 48), color=(10, 20, 30)).save(input_path)
    standardize_and_save(input_path=input_path, output_path=output_path, size=(16, 16))

    assert output_path.exists()
    saved = load_image(output_path)
    assert saved.size == (16, 16)
    assert saved.format is None  # copy() detaches format metadata


def test_save_png_and_list_pngs(tmp_path: Path) -> None:
    out_dir = tmp_path / "images"
    a = out_dir / "a.png"
    b = out_dir / "b.png"
    c = out_dir / "c.jpg"

    save_png(Image.new("RGB", (4, 4), color=(1, 2, 3)), a)
    save_png(Image.new("RGB", (4, 4), color=(4, 5, 6)), b)
    Image.new("RGB", (4, 4), color=(7, 8, 9)).save(c)

    assert list(list_pngs(out_dir)) == [a, b]
