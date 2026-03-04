from __future__ import annotations

import json
from pathlib import Path

from src.data.manifests import write_rows_csv
from src.pipeline.cli import _resolve_path, main
from tests.helpers import STEGO_FIELDNAMES, write_cover_manifest


def test_resolve_path_relative_and_absolute(tmp_path: Path) -> None:
    project_root = tmp_path
    relative = Path("data/x.csv")
    absolute = tmp_path / "abs.csv"

    assert _resolve_path(relative, project_root) == project_root / relative
    assert _resolve_path(absolute, project_root) == absolute


def test_cli_init_layout(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--project-root", str(tmp_path), "init-layout"],
    )
    main()
    out = capsys.readouterr().out

    assert "Layout initialized" in out
    assert (tmp_path / "data" / "covers" / "real").is_dir()
    assert (tmp_path / "results" / "metrics").is_dir()


def test_cli_build_training_jobs(monkeypatch, capsys, tmp_path: Path) -> None:
    splits_json = tmp_path / "results" / "splits" / "splits_grouped5fold.json"
    splits_json.parent.mkdir(parents=True, exist_ok=True)

    folds = []
    for i in range(5):
        base = i * 100
        test = list(range(base + 1, base + 101))
        val = list(range(401, 451))
        train = [g for g in range(1, 501) if g not in set(test) and g not in set(val)]
        folds.append(
            {
                "fold": i,
                "train_group_ids": train,
                "val_group_ids": val,
                "test_group_ids": test,
            }
        )
    splits_json.write_text(
        json.dumps({"protocol": "grouped-5fold", "group_unit": "group_id", "folds": folds}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--project-root",
            str(tmp_path),
            "build-training-jobs",
            "--splits-json",
            "results/splits/splits_grouped5fold.json",
        ],
    )
    main()
    out = capsys.readouterr().out

    assert "Training jobs CSV" in out
    jobs_path = tmp_path / "results" / "splits" / "srm_training_jobs.csv"
    assert jobs_path.exists()


def test_cli_run_embedding_stage_dry_run(monkeypatch, capsys, tmp_path: Path) -> None:
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
                "cover_path": "/tmp/cover.png",
                "payload_path": "/tmp/payload.bin",
                "stego_path": "/tmp/stego.png",
                "embed_params": "{}",
                "seed": "42",
            }
        ],
        fieldnames=STEGO_FIELDNAMES,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--project-root",
            str(tmp_path),
            "run-embedding-stage",
            "--stego-manifest",
            "data/manifests/stego_manifest.csv",
        ],
    )
    main()
    out = capsys.readouterr().out

    assert "Embedding rows processed: 1" in out


def test_cli_create_splits(monkeypatch, capsys, tmp_path: Path) -> None:
    covers_manifest = write_cover_manifest(
        tmp_path / "data" / "manifests" / "covers_master.csv",
        group_ids=range(1, 501),
    )
    assert covers_manifest.exists()

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--project-root",
            str(tmp_path),
            "create-splits",
            "--covers-manifest",
            "data/manifests/covers_master.csv",
        ],
    )
    main()
    out = capsys.readouterr().out

    assert "Splits JSON" in out
    assert (tmp_path / "results" / "splits" / "splits_grouped5fold.json").exists()
