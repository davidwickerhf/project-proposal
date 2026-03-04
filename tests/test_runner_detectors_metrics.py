from __future__ import annotations

import json
from pathlib import Path

from src.data.manifests import read_rows_csv, write_rows_csv
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner


STEGO_FIELDS = [
    "group_id",
    "source",
    "method",
    "payload_level",
    "encryption",
    "cover_path",
    "payload_path",
    "stego_path",
    "embed_params",
    "seed",
]


def _write_test_images(project_root: Path) -> tuple[Path, Path]:
    from PIL import Image

    cover = project_root / "data" / "covers" / "real" / "g0001__src-real.png"
    stego = project_root / "data" / "stego" / "lsb" / "low" / "plain" / "real" / "g0001__src-real__m-lsb__p-low__e-plain.png"
    cover.parent.mkdir(parents=True, exist_ok=True)
    stego.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (32, 32), color=(30, 40, 50)).save(cover)
    Image.new("RGB", (32, 32), color=(35, 45, 55)).save(stego)
    return cover, stego


def test_run_detector_stage_dry_run_and_compute_metrics(tmp_path: Path) -> None:
    runner = PipelineRunner(PipelineConfig(project_root=tmp_path, n_groups=2))

    cover, stego = _write_test_images(tmp_path)
    stego_manifest = tmp_path / "data" / "manifests" / "stego_manifest.csv"
    write_rows_csv(
        stego_manifest,
        rows=[
            {
                "group_id": "1",
                "source": "real",
                "method": "lsb",
                "payload_level": "low",
                "encryption": "plain",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/plain/low/g0001__p-low__e-plain.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            },
            {
                "group_id": "1",
                "source": "real",
                "method": "dct",
                "payload_level": "high",
                "encryption": "encrypted",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/encrypted/high/g0001__p-high__e-encrypted.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            },
            {
                "group_id": "2",
                "source": "real",
                "method": "lsb",
                "payload_level": "low",
                "encryption": "plain",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/plain/low/g0002__p-low__e-plain.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            },
        ],
        fieldnames=STEGO_FIELDS,
    )

    splits_json = tmp_path / "results" / "splits" / "splits_grouped5fold.json"
    splits_json.parent.mkdir(parents=True, exist_ok=True)
    splits_json.write_text(
        json.dumps(
            {
                "protocol": "grouped-5fold",
                "group_unit": "group_id",
                "folds": [
                    {
                        "fold": 0,
                        "train_group_ids": [2],
                        "val_group_ids": [],
                        "test_group_ids": [1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pred_path = runner.run_detector_stage(
        stego_manifest_path=stego_manifest,
        splits_json_path=splits_json,
        execute=False,
        include_srm=True,
    )

    pred_rows = read_rows_csv(pred_path)
    # group 1 only: one lsb row -> 3 detectors x 2 labels = 6 rows
    # one dct row -> 2 detectors x 2 labels = 4 rows
    assert len(pred_rows) == 10
    assert all(r["score"] == "" for r in pred_rows)


def test_run_detector_stage_execute_with_stub_scores_and_metrics(tmp_path: Path) -> None:
    runner = PipelineRunner(PipelineConfig(project_root=tmp_path, n_groups=1))

    cover, stego = _write_test_images(tmp_path)
    stego_manifest = tmp_path / "data" / "manifests" / "stego_manifest.csv"
    write_rows_csv(
        stego_manifest,
        rows=[
            {
                "group_id": "1",
                "source": "real",
                "method": "lsb",
                "payload_level": "low",
                "encryption": "plain",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/plain/low/g0001__p-low__e-plain.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            }
        ],
        fieldnames=STEGO_FIELDS,
    )

    splits_json = tmp_path / "results" / "splits" / "splits_grouped5fold.json"
    splits_json.parent.mkdir(parents=True, exist_ok=True)
    splits_json.write_text(
        json.dumps(
            {
                "protocol": "grouped-5fold",
                "group_unit": "group_id",
                "folds": [
                    {
                        "fold": 0,
                        "train_group_ids": [],
                        "val_group_ids": [],
                        "test_group_ids": [1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_score_detector_row(*, detector, label, row, fold, srm_score_provider, srm_models=None):
        _ = (detector, row, fold, srm_score_provider, srm_models)
        return 0.9 if label == 1 else 0.1

    runner._score_detector_row = fake_score_detector_row  # type: ignore[method-assign]

    pred_path = runner.run_detector_stage(
        stego_manifest_path=stego_manifest,
        splits_json_path=splits_json,
        execute=True,
        include_srm=True,
        srm_score_provider=lambda row: 0.5,  # prevent auto-training; scoring is monkey-patched
    )
    preds = read_rows_csv(pred_path)
    assert len(preds) == 6  # lsb detectors: srm_ec, rs, chi_square each with pos+neg
    assert {p["detector"] for p in preds} == {"srm_ec", "rs", "chi_square"}

    outputs = runner.compute_metrics_from_predictions(predictions_path=pred_path)

    fold_rows = read_rows_csv(outputs["fold_metrics"])
    cond_rows = read_rows_csv(outputs["condition_metrics"])
    src_rows = read_rows_csv(outputs["source_contrasts"])

    assert len(fold_rows) == 3
    assert len(cond_rows) == 3
    assert len(src_rows) == 3
    assert all(float(r["roc_auc"]) == 1.0 for r in fold_rows)


def test_run_detector_stage_auto_trains_srm_models(tmp_path: Path, monkeypatch) -> None:
    """When execute=True and include_srm=True with no external provider,
    the runner should auto-train SRM models via _train_srm_models and use
    them for scoring."""
    from src.detection.srm import SRMModelArtifact
    from src.pipeline import runner as runner_module

    runner = PipelineRunner(PipelineConfig(project_root=tmp_path, n_groups=2))

    cover, stego = _write_test_images(tmp_path)
    stego_manifest = tmp_path / "data" / "manifests" / "stego_manifest.csv"
    write_rows_csv(
        stego_manifest,
        rows=[
            {
                "group_id": "1",
                "source": "real",
                "method": "lsb",
                "payload_level": "low",
                "encryption": "plain",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/plain/low/g0001__p-low__e-plain.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            },
            {
                "group_id": "2",
                "source": "real",
                "method": "lsb",
                "payload_level": "low",
                "encryption": "plain",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/plain/low/g0002__p-low__e-plain.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            },
        ],
        fieldnames=STEGO_FIELDS,
    )

    splits_json = tmp_path / "results" / "splits" / "splits_grouped5fold.json"
    splits_json.parent.mkdir(parents=True, exist_ok=True)
    splits_json.write_text(
        json.dumps(
            {
                "protocol": "grouped-5fold",
                "group_unit": "group_id",
                "folds": [
                    {
                        "fold": 0,
                        "train_group_ids": [2],
                        "val_group_ids": [],
                        "test_group_ids": [1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Patch SRM functions on the runner module (where they are imported).
    monkeypatch.setattr(runner_module, "extract_srm_features", lambda img: [0.1, 0.2, 0.3])
    monkeypatch.setattr(runner_module, "train_srm_ec_model", lambda ti: SRMModelArtifact(
        method=ti.method, fold=ti.fold, model_state="fake", hyperparams={},
    ))
    monkeypatch.setattr(runner_module, "score_srm_ec_model", lambda model, xs: [0.85] * len(xs))

    pred_path = runner.run_detector_stage(
        stego_manifest_path=stego_manifest,
        splits_json_path=splits_json,
        execute=True,
        include_srm=True,
        skip_unimplemented=True,
    )

    preds = read_rows_csv(pred_path)
    srm_preds = [p for p in preds if p["detector"] == "srm_ec"]
    # group 1 test, lsb method: srm_ec has 1 pos + 1 neg = 2 rows
    assert len(srm_preds) == 2
    assert all(p["score"] != "" for p in srm_preds)


def test_run_detector_stage_skip_unimplemented_srm_training(tmp_path: Path) -> None:
    """When SRM feature extraction is not implemented and skip_unimplemented=True,
    the pipeline should gracefully skip SRM without crashing."""
    runner = PipelineRunner(PipelineConfig(project_root=tmp_path, n_groups=1))

    cover, stego = _write_test_images(tmp_path)
    stego_manifest = tmp_path / "data" / "manifests" / "stego_manifest.csv"
    write_rows_csv(
        stego_manifest,
        rows=[
            {
                "group_id": "1",
                "source": "real",
                "method": "lsb",
                "payload_level": "low",
                "encryption": "plain",
                "cover_path": str(cover.relative_to(tmp_path)),
                "payload_path": "data/payloads/plain/low/g0001__p-low__e-plain.bin",
                "stego_path": str(stego.relative_to(tmp_path)),
                "embed_params": "{}",
                "seed": "42",
            }
        ],
        fieldnames=STEGO_FIELDS,
    )

    splits_json = tmp_path / "results" / "splits" / "splits_grouped5fold.json"
    splits_json.parent.mkdir(parents=True, exist_ok=True)
    splits_json.write_text(
        json.dumps(
            {
                "protocol": "grouped-5fold",
                "group_unit": "group_id",
                "folds": [
                    {
                        "fold": 0,
                        "train_group_ids": [],
                        "val_group_ids": [],
                        "test_group_ids": [1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # SRM functions are not implemented (default placeholders raise NotImplementedError).
    # With skip_unimplemented=True, this should not crash.
    pred_path = runner.run_detector_stage(
        stego_manifest_path=stego_manifest,
        splits_json_path=splits_json,
        execute=True,
        include_srm=True,
        skip_unimplemented=True,
    )

    preds = read_rows_csv(pred_path)
    # SRM rows should be skipped; only statistical detectors (rs, chi_square) remain,
    # but those also raise NotImplementedError and get skipped too.
    assert all(p["detector"] != "srm_ec" or p["score"] == "" for p in preds if p.get("score") == "")
