from __future__ import annotations

from pathlib import Path

from src.data.manifests import read_rows_csv, write_rows_csv


def test_generate_ml_covers_stub_creates_images_and_manifests(tmp_path: Path) -> None:
    from src.data.generate_ml_covers import generate_ml_covers_from_prompts

    prompts_path = tmp_path / "data" / "manifests" / "generation_prompts.csv"
    prompts_rows = [
        {
            "group_id": "1",
            "dataset": "COCO",
            "orig_id": "orig-1",
            "caption_id": "cap-1",
            "caption_text": "A person riding a bicycle through a busy city street.",
            "real_image_path": "data/covers/real/g0001__src-real.png",
        },
        {
            "group_id": "2",
            "dataset": "Flickr30k",
            "orig_id": "orig-2",
            "caption_id": "cap-2",
            "caption_text": "Two children playing football near a metal fence in a park.",
            "real_image_path": "data/covers/real/g0002__src-real.png",
        },
        {
            "group_id": "3",
            "dataset": "COCO",
            "orig_id": "orig-3",
            "caption_id": "cap-3",
            "caption_text": "A chef preparing food in a restaurant kitchen.",
            "real_image_path": "data/covers/real/g0003__src-real.png",
        },
    ]
    write_rows_csv(
        prompts_path,
        prompts_rows,
        fieldnames=[
            "group_id",
            "dataset",
            "orig_id",
            "caption_id",
            "caption_text",
            "real_image_path",
        ],
    )

    outputs = generate_ml_covers_from_prompts(
        project_root=tmp_path,
        prompts_csv=prompts_path,
        engine="stub",
        image_size=(512, 512),
        width=256,
        height=256,
        seed_base=123,
    )

    a_rows = read_rows_csv(outputs["covers_master_ml_a"])
    b_rows = read_rows_csv(outputs["covers_master_ml_b"])
    ml_rows = read_rows_csv(outputs["covers_master_ml"])

    assert len(a_rows) == 3
    assert len(b_rows) == 3
    assert len(ml_rows) == 6

    assert {r["source"] for r in a_rows} == {"ml_a"}
    assert {r["source"] for r in b_rows} == {"ml_b"}
    assert {r["dataset"] for r in a_rows} == {"SDXL"}
    assert {r["dataset"] for r in b_rows} == {"PixArt-alpha"}

    for row in ml_rows:
        out_path = tmp_path / row["image_path"]
        assert out_path.exists()
        assert out_path.name.startswith("g") and out_path.name.endswith(".png")
        assert not Path(row["image_path"]).is_absolute()


def test_generate_ml_covers_stub_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    from src.data.generate_ml_covers import generate_ml_covers_from_prompts

    prompts_path = tmp_path / "data" / "manifests" / "generation_prompts.csv"
    write_rows_csv(
        prompts_path,
        rows=[
            {
                "group_id": "1",
                "dataset": "COCO",
                "orig_id": "orig-1",
                "caption_id": "cap-1",
                "caption_text": "A red bus driving through downtown traffic.",
                "real_image_path": "data/covers/real/g0001__src-real.png",
            }
        ],
        fieldnames=[
            "group_id",
            "dataset",
            "orig_id",
            "caption_id",
            "caption_text",
            "real_image_path",
        ],
    )

    generate_ml_covers_from_prompts(
        project_root=tmp_path,
        prompts_csv=prompts_path,
        engine="stub",
        width=128,
        height=128,
        image_size=(512, 512),
        seed_base=77,
    )
    first_bytes = (tmp_path / "data/covers/ml_a/g0001__src-ml_a.png").read_bytes()

    generate_ml_covers_from_prompts(
        project_root=tmp_path,
        prompts_csv=prompts_path,
        engine="stub",
        width=128,
        height=128,
        image_size=(512, 512),
        seed_base=77,
    )
    second_bytes = (tmp_path / "data/covers/ml_a/g0001__src-ml_a.png").read_bytes()

    assert first_bytes == second_bytes
