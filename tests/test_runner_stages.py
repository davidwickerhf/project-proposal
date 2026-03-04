from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.manifests import read_rows_csv, write_rows_csv
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner, _stable_iv
from tests.helpers import COVER_FIELDNAMES, create_image, write_cover_manifest


def test_standardize_covers_from_index_writes_outputs(project_root: Path, runner: PipelineRunner) -> None:
    raw_root = project_root / "raw"
    index_csv = project_root / "raw_index.csv"

    rows: list[dict[str, str]] = []
    for group_id in [1, 2]:
        for source in ["real", "ml_a", "ml_b"]:
            raw_path = create_image(raw_root / f"{group_id}_{source}.jpg", size=(23, 17))
            rows.append(
                {
                    "group_id": str(group_id),
                    "source": source,
                    "dataset": "fixture",
                    "orig_id": f"orig-{group_id}",
                    "caption_id": f"cap-{group_id}",
                    "caption_text": f"caption {group_id}",
                    "raw_image_path": str(raw_path),
                    "qc_pass": "true",
                    "qc_score": "0.9",
                    "seed": "123",
                }
            )

    write_rows_csv(
        index_csv,
        rows,
        fieldnames=[
            "group_id",
            "source",
            "dataset",
            "orig_id",
            "caption_id",
            "caption_text",
            "raw_image_path",
            "qc_pass",
            "qc_score",
            "seed",
        ],
    )

    out = runner.standardize_covers_from_index(index_csv)
    covers = read_rows_csv(out)

    assert len(covers) == 6
    first_path = Path(covers[0]["image_path"])
    assert first_path.exists()

    from src.data.images import load_image

    standardized = load_image(first_path)
    assert standardized.size == (512, 512)


def test_create_grouped_splits_and_training_jobs_locked_design(project_root: Path) -> None:
    cfg = PipelineConfig(project_root=project_root)
    local_runner = PipelineRunner(cfg)

    covers_manifest = write_cover_manifest(
        project_root / "data" / "manifests" / "covers_master.csv",
        group_ids=range(1, 501),
    )

    splits_json = local_runner.create_grouped_splits(covers_manifest_path=covers_manifest)
    split_obj = json.loads(splits_json.read_text(encoding="utf-8"))

    assert split_obj["protocol"] == "grouped-5fold"
    assert len(split_obj["folds"]) == 5

    for fold in split_obj["folds"]:
        assert len(fold["train_group_ids"]) == 350
        assert len(fold["val_group_ids"]) == 50
        assert len(fold["test_group_ids"]) == 100

    jobs_csv = local_runner.build_srm_training_jobs(splits_json_path=splits_json)
    jobs = read_rows_csv(jobs_csv)

    assert len(jobs) == 10
    assert {j["method"] for j in jobs} == {"lsb", "dct"}
    assert {j["train_samples"] for j in jobs} == {"6300"}
    assert {j["val_samples"] for j in jobs} == {"900"}
    assert {j["test_samples"] for j in jobs} == {"1800"}


def test_build_payload_manifest_validates_group_count(project_root: Path, runner: PipelineRunner) -> None:
    covers_manifest = write_cover_manifest(
        project_root / "data" / "manifests" / "covers_master.csv",
        group_ids=[1, 2, 3],
    )

    with pytest.raises(ValueError, match="Expected 500 groups"):
        runner.build_payload_manifest(covers_manifest_path=covers_manifest)


def test_stable_iv_is_deterministic_and_16_bytes() -> None:
    iv_a = _stable_iv(10, "medium")
    iv_b = _stable_iv(10, "medium")
    iv_c = _stable_iv(10, "low")

    assert iv_a == iv_b
    assert iv_a != iv_c
    assert isinstance(iv_a, bytes)
    assert len(iv_a) == 16


def test_embed_params_json_contract(project_root: Path) -> None:
    runner = PipelineRunner(PipelineConfig(project_root=project_root, n_groups=4))

    lsb_low = json.loads(runner._embed_params_json("lsb", "low"))
    assert lsb_low["method"] == "lsb"
    assert lsb_low["k"] == 1
    assert lsb_low["pixel_fraction"] == 0.25

    dct_high = json.loads(runner._embed_params_json("dct", "high"))
    assert dct_high["method"] == "dct_qim"
    assert dct_high["coeff_fraction"] == 0.50

    with pytest.raises(ValueError, match="Unknown method"):
        runner._embed_params_json("foo", "low")
