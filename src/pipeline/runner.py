from __future__ import annotations

"""Stage orchestrator for the steganography experiment pipeline.

`PipelineRunner` is the only layer that should handle file I/O for deferred
algorithm components. Deferred embedding/encryption/detection functions are
kept closed-loop (in-memory in/out), while this module:
- reads/writes manifests,
- resolves relative paths,
- materializes artifacts in canonical layout.
"""

import hashlib
import json
import secrets
from pathlib import Path
import random
from typing import Callable

from src.common.contracts import ENCRYPTION_STATES, METHODS, PAYLOAD_LEVELS
from src.data.manifests import (
    CoverRecord,
    PayloadRecord,
    StegoRecord,
    TrainingJobRecord,
    read_rows_csv,
    unique_group_ids,
    write_dataclass_csv,
    write_rows_csv,
    write_json,
)
from src.data.images import load_image, standardize_and_save
from src.detection.statistical import (
    block_dct_shift_score,
    chi_square_score,
    rs_analysis_score,
)
from src.embedding.dct import embed_dct_qim
from src.embedding.encryption import encrypt_payload_aes_256_cbc
from src.embedding.lsb import embed_lsb
from src.evaluation.metrics import (
    aggregate_by_groups,
    summarize_fold_mean_interval,
    try_parse_score,
)
from src.evaluation.plots import generate_metrics_figures
from src.evaluation.splits import FoldSplit, generate_grouped_5fold_splits
from src.pipeline.config import PipelineConfig


def _stable_iv(group_id: int, payload_level: str) -> bytes:
    """Create a deterministic 16-byte IV from group and payload level."""
    digest = hashlib.sha256(f"{group_id}:{payload_level}".encode("utf-8")).digest()
    return digest[:16]


class PipelineRunner:
    """Execute pipeline stages and keep all contracts aligned with README."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.paths = config.paths

    def init_layout(self) -> None:
        """Create the full expected directory hierarchy for data/results artifacts."""
        self.paths.ensure_layout()

    def _resolve_manifest_path(self, value: str | Path) -> Path:
        """Resolve relative manifest paths against project root for local file access."""
        path = Path(value)
        return path if path.is_absolute() else (self.config.project_root / path)

    def _to_project_relative(self, value: Path | str) -> str:
        """Store manifest paths relative to project root when possible."""
        path = self._resolve_manifest_path(value)
        try:
            return str(path.relative_to(self.config.project_root))
        except ValueError:
            # If a path is outside project root, keep absolute path as fallback.
            return str(path)

    def _load_folds(self, splits_json_path: Path) -> list[FoldSplit]:
        obj = json.loads(self._resolve_manifest_path(splits_json_path).read_text(encoding="utf-8"))
        return [FoldSplit(**fold) for fold in obj["folds"]]

    def _detectors_for_method(self, method: str, include_srm: bool) -> list[str]:
        if method == "lsb":
            detectors = ["rs", "chi_square"]
        elif method == "dct":
            detectors = ["block_dct_shift"]
        else:
            raise ValueError(f"Unknown method: {method}")
        if include_srm:
            detectors.insert(0, "srm_ec")
        return detectors

    def standardize_covers_from_index(
        self,
        input_index_csv: Path,
        output_manifest_path: Path | None = None,
    ) -> Path:
        """Standardize raw cover images into canonical PNG storage.

        Required input columns:
        group_id, source, dataset, orig_id, caption_id, caption_text,
        raw_image_path, qc_pass, qc_score, seed
        """
        rows = read_rows_csv(input_index_csv)
        records: list[CoverRecord] = []
        for row in rows:
            group_id = int(row["group_id"])
            source = row["source"]
            out_path = self.paths.cover_path(group_id, source)  # type: ignore[arg-type]
            # Read raw image path from manifest, standardize to canonical cover format.
            standardize_and_save(
                input_path=self._resolve_manifest_path(row["raw_image_path"]),
                output_path=out_path,
                size=self.config.image_size,
            )
            records.append(
                CoverRecord(
                    group_id=group_id,
                    source=source,
                    dataset=row["dataset"],
                    orig_id=row["orig_id"],
                    caption_id=row["caption_id"],
                    caption_text=row["caption_text"],
                    image_path=self._to_project_relative(out_path),
                    qc_pass=row["qc_pass"].lower() == "true",
                    qc_score=float(row["qc_score"]),
                    seed=int(row["seed"]),
                )
            )

        # Covers manifest is the source-of-truth for all downstream pairing.
        output_path = output_manifest_path or (self.paths.manifests_dir / "covers_master.csv")
        write_dataclass_csv(output_path, records)
        return output_path

    def build_payload_manifest(
        self,
        covers_manifest_path: Path,
        output_manifest_path: Path | None = None,
        write_payload_files: bool = False,
    ) -> Path:
        """Generate payload manifest entries for all groups/levels/encryption states.

        When ``write_payload_files`` is True, payload binaries are also materialized.
        Encryption branch intentionally calls the placeholder AES function.
        """
        cover_rows = read_rows_csv(covers_manifest_path)
        groups = unique_group_ids(cover_rows)
        if len(groups) != self.config.n_groups:
            raise ValueError(
                f"Expected {self.config.n_groups} groups in covers manifest, got {len(groups)}."
            )

        rng = random.Random(self.config.payload_seed)
        records: list[PayloadRecord] = []
        for group_id in groups:
            for payload_level in PAYLOAD_LEVELS:
                payload_bits = self.config.payload_bits_by_level[payload_level]
                payload_bytes = self._generate_payload_bytes(payload_bits, rng)
                for encryption in ENCRYPTION_STATES:
                    payload_path = self.paths.payload_path(group_id, payload_level, encryption)
                    iv = _stable_iv(group_id, payload_level)

                    if write_payload_files:
                        # Optional artifact materialization for fully executable runs.
                        payload_path.parent.mkdir(parents=True, exist_ok=True)
                        if encryption == "plain":
                            payload_path.write_bytes(payload_bytes)
                        else:
                            # Intentionally calls placeholder implementation.
                            key = secrets.token_bytes(32)
                            ciphertext = encrypt_payload_aes_256_cbc(payload_bytes, key=key, iv=iv)
                            payload_path.write_bytes(ciphertext)

                    records.append(
                        PayloadRecord(
                            group_id=group_id,
                            payload_level=payload_level,
                            encryption=encryption,
                            payload_path=self._to_project_relative(payload_path),
                            payload_bits=payload_bits,
                            aes_iv=iv.hex(),
                            aes_key_id=self.config.aes_key_id,
                            seed=self.config.payload_seed,
                        )
                    )

        output_path = output_manifest_path or (self.paths.manifests_dir / "payload_manifest.csv")
        write_dataclass_csv(output_path, records)
        return output_path

    def build_stego_manifest(
        self,
        covers_manifest_path: Path,
        payload_manifest_path: Path,
        output_manifest_path: Path | None = None,
    ) -> Path:
        """Enumerate all stego jobs from covers x methods x payloads x encryption."""
        cover_rows = read_rows_csv(covers_manifest_path)
        payload_rows = read_rows_csv(payload_manifest_path)

        # Fast lookup by the natural composite key of payload artifacts.
        payload_index = {
            (int(row["group_id"]), row["payload_level"], row["encryption"]): row
            for row in payload_rows
        }

        records: list[StegoRecord] = []
        for cover in cover_rows:
            group_id = int(cover["group_id"])
            source = cover["source"]
            cover_path = self._to_project_relative(cover["image_path"])
            for method in METHODS:
                for payload_level in PAYLOAD_LEVELS:
                    for encryption in ENCRYPTION_STATES:
                        payload_row = payload_index[(group_id, payload_level, encryption)]
                        payload_path = self._to_project_relative(payload_row["payload_path"])
                        stego_path = self.paths.stego_path(
                            group_id=group_id,
                            source=source,  # type: ignore[arg-type]
                            method=method,  # type: ignore[arg-type]
                            payload=payload_level,  # type: ignore[arg-type]
                            encryption=encryption,  # type: ignore[arg-type]
                        )
                        embed_params = self._embed_params_json(method, payload_level)
                        records.append(
                            StegoRecord(
                                group_id=group_id,
                                source=source,
                                method=method,
                                payload_level=payload_level,
                                encryption=encryption,
                                cover_path=cover_path,
                                payload_path=payload_path,
                                stego_path=self._to_project_relative(stego_path),
                                embed_params=embed_params,
                                seed=self.config.embed_seed,
                            )
                        )

        # This manifest drives the embedding execution stage.
        output_path = output_manifest_path or (self.paths.manifests_dir / "stego_manifest.csv")
        write_dataclass_csv(output_path, records)
        return output_path

    def create_grouped_splits(
        self,
        covers_manifest_path: Path,
        output_path: Path | None = None,
    ) -> Path:
        """Create grouped 5-fold split JSON from the covers manifest group IDs."""
        cover_rows = read_rows_csv(covers_manifest_path)
        group_ids = unique_group_ids(cover_rows)
        folds = generate_grouped_5fold_splits(
            group_ids=group_ids,
            seed=self.config.split_seed,
            val_groups_per_fold=50,
        )
        split_payload = {
            "protocol": "grouped-5fold",
            "group_unit": "group_id",
            "folds": [fold.to_dict() for fold in folds],
        }
        path = output_path or (self.paths.splits_dir / "splits_grouped5fold.json")
        write_json(path, split_payload)
        return path

    def build_srm_training_jobs(
        self,
        splits_json_path: Path,
        output_manifest_path: Path | None = None,
    ) -> Path:
        """Build per-method SRM training job manifest from split definitions."""
        splits_obj = json.loads(splits_json_path.read_text(encoding="utf-8"))
        folds = [FoldSplit(**fold) for fold in splits_obj["folds"]]

        records: list[TrainingJobRecord] = []
        for fold in folds:
            for method in METHODS:
                train_groups = len(fold.train_group_ids)
                val_groups = len(fold.val_group_ids)
                test_groups = len(fold.test_group_ids)
                records.append(
                    TrainingJobRecord(
                        fold=fold.fold,
                        method=method,
                        train_groups=train_groups,
                        val_groups=val_groups,
                        test_groups=test_groups,
                        train_samples=train_groups * 3 * 6,
                        val_samples=val_groups * 3 * 6,
                        test_samples=test_groups * 3 * 6,
                        split_ref=self._to_project_relative(splits_json_path),
                    )
                )

        # 2 methods x 5 folds = 10 jobs in the locked design.
        output_path = output_manifest_path or (self.paths.splits_dir / "srm_training_jobs.csv")
        write_dataclass_csv(output_path, records)
        return output_path

    def run_embedding_stage(
        self,
        stego_manifest_path: Path,
        execute: bool = False,
    ) -> int:
        """Create stego artifacts from manifest rows.

        With execute=False this is a dry-run counter only.
        """
        rows = read_rows_csv(stego_manifest_path)
        if not execute:
            return len(rows)

        # This stage intentionally calls placeholder methods.
        from src.data.images import save_png

        for row in rows:
            # Runner handles file I/O; embedding functions stay closed-loop.
            cover_image = load_image(self._resolve_manifest_path(row["cover_path"]))
            payload_bytes = self._resolve_manifest_path(row["payload_path"]).read_bytes()
            method = row["method"]
            payload_level = row["payload_level"]
            if method == "lsb":
                stego = embed_lsb(
                    cover_image=cover_image,
                    payload_bytes=payload_bytes,
                    payload_level=payload_level,
                    prng_key=self.config.lsb_prng_key,
                )
            elif method == "dct":
                stego = embed_dct_qim(
                    cover_image=cover_image,
                    payload_bytes=payload_bytes,
                    payload_level=payload_level,
                    delta=self.config.dct_delta,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            save_png(stego, self._resolve_manifest_path(row["stego_path"]))
        return len(rows)

    def run_detector_stage(
        self,
        stego_manifest_path: Path,
        splits_json_path: Path,
        output_path: Path | None = None,
        *,
        execute: bool = False,
        include_srm: bool = True,
        skip_unimplemented: bool = False,
        srm_score_provider: Callable[[dict[str, str]], float] | None = None,
    ) -> Path:
        """Run detector scoring on fold test partitions and write predictions CSV.

        Output schema:
        fold, detector, group_id, source, method, payload_level, encryption, label, score

        Notes:
        - Positives are stego rows (label=1).
        - Negatives are matched cover rows per stego condition (label=0).
        - With execute=False, score values are left empty as a dry-run plan.
        """
        stego_rows = read_rows_csv(stego_manifest_path)
        folds = self._load_folds(splits_json_path)

        pred_rows: list[dict[str, object]] = []
        for fold in folds:
            test_groups = set(fold.test_group_ids)
            for row in stego_rows:
                group_id = int(row["group_id"])
                if group_id not in test_groups:
                    continue

                method = row["method"]
                detectors = self._detectors_for_method(method, include_srm=include_srm)

                for detector in detectors:
                    # Positive row (stego image).
                    pos_score = ""
                    if execute:
                        try:
                            pos_score = self._score_detector_row(
                                detector=detector,
                                label=1,
                                row=row,
                                srm_score_provider=srm_score_provider,
                            )
                        except NotImplementedError:
                            if skip_unimplemented:
                                continue
                            raise

                    pred_rows.append(
                        {
                            "fold": fold.fold,
                            "detector": detector,
                            "group_id": group_id,
                            "source": row["source"],
                            "method": method,
                            "payload_level": row["payload_level"],
                            "encryption": row["encryption"],
                            "label": 1,
                            "score": pos_score,
                        }
                    )

                    # Matched negative row (cover image under same condition key).
                    neg_score = ""
                    if execute:
                        try:
                            neg_score = self._score_detector_row(
                                detector=detector,
                                label=0,
                                row=row,
                                srm_score_provider=srm_score_provider,
                            )
                        except NotImplementedError:
                            if skip_unimplemented:
                                # Remove positive row written above to keep label balance for this detector.
                                pred_rows.pop()
                                continue
                            raise

                    pred_rows.append(
                        {
                            "fold": fold.fold,
                            "detector": detector,
                            "group_id": group_id,
                            "source": row["source"],
                            "method": method,
                            "payload_level": row["payload_level"],
                            "encryption": row["encryption"],
                            "label": 0,
                            "score": neg_score,
                        }
                    )

        out = output_path or (self.paths.predictions_dir / "predictions.csv")
        write_rows_csv(
            out,
            pred_rows,
            fieldnames=[
                "fold",
                "detector",
                "group_id",
                "source",
                "method",
                "payload_level",
                "encryption",
                "label",
                "score",
            ],
        )
        return out

    def compute_metrics_from_predictions(
        self,
        predictions_path: Path,
        metrics_dir: Path | None = None,
        quality_metrics_input: Path | None = None,
    ) -> dict[str, Path]:
        """Compute fold/condition/source metrics from detector prediction rows.

        This stage computes:
        - fold_metrics.csv
        - condition_metrics.csv
        - source_contrasts.csv
        - pooled_summary.csv
        - quality_metrics.csv (copied from input, or empty scaffold file)
        """
        rows = read_rows_csv(predictions_path)
        # Keep only rows with numeric scores for metric calculations.
        scored_rows = [r for r in rows if try_parse_score(r.get("score", "")) is not None]

        out_dir = metrics_dir or self.paths.metrics_dir
        out_dir = self._resolve_manifest_path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fold_metrics = aggregate_by_groups(scored_rows, ["fold", "detector"])
        condition_metrics = aggregate_by_groups(
            scored_rows, ["fold", "detector", "method", "payload_level", "encryption"]
        )
        source_metrics = aggregate_by_groups(scored_rows, ["fold", "detector", "source"])
        pooled_summary = summarize_fold_mean_interval(
            source_metrics,
            metric_key="roc_auc",
            group_keys_excluding_fold=["detector", "source"],
        )

        fold_path = out_dir / "fold_metrics.csv"
        condition_path = out_dir / "condition_metrics.csv"
        source_path = out_dir / "source_contrasts.csv"
        pooled_path = out_dir / "pooled_summary.csv"
        quality_path = out_dir / "quality_metrics.csv"

        write_rows_csv(
            fold_path,
            fold_metrics,
            fieldnames=[
                "fold",
                "detector",
                "n_samples",
                "n_pos",
                "n_neg",
                "roc_auc",
                "eer",
                "accuracy_at_youden_j",
                "fpr_at_fixed_fnr",
            ],
        )
        write_rows_csv(
            condition_path,
            condition_metrics,
            fieldnames=[
                "fold",
                "detector",
                "method",
                "payload_level",
                "encryption",
                "n_samples",
                "n_pos",
                "n_neg",
                "roc_auc",
                "eer",
                "accuracy_at_youden_j",
                "fpr_at_fixed_fnr",
            ],
        )
        write_rows_csv(
            source_path,
            source_metrics,
            fieldnames=[
                "fold",
                "detector",
                "source",
                "n_samples",
                "n_pos",
                "n_neg",
                "roc_auc",
                "eer",
                "accuracy_at_youden_j",
                "fpr_at_fixed_fnr",
            ],
        )
        write_rows_csv(
            pooled_path,
            pooled_summary,
            fieldnames=["detector", "source", "roc_auc_mean", "roc_auc_std", "n_folds"],
        )

        if quality_metrics_input is not None:
            in_path = self._resolve_manifest_path(quality_metrics_input)
            write_rows_csv(
                quality_path,
                read_rows_csv(in_path),
                fieldnames=[
                    "group_id",
                    "source",
                    "method",
                    "payload_level",
                    "encryption",
                    "psnr",
                    "ssim",
                    "fsim",
                ],
            )
        else:
            write_rows_csv(
                quality_path,
                [],
                fieldnames=[
                    "group_id",
                    "source",
                    "method",
                    "payload_level",
                    "encryption",
                    "psnr",
                    "ssim",
                    "fsim",
                ],
            )

        return {
            "fold_metrics": fold_path,
            "condition_metrics": condition_path,
            "source_contrasts": source_path,
            "pooled_summary": pooled_path,
            "quality_metrics": quality_path,
        }

    def generate_metrics_figures(
        self,
        metrics_dir: Path | None = None,
        figures_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Generate core metric figures from metrics CSV outputs."""
        resolved_metrics_dir = self._resolve_manifest_path(metrics_dir or self.paths.metrics_dir)
        resolved_figures_dir = self._resolve_manifest_path(
            figures_dir or self.paths.figures_dir
        )
        return generate_metrics_figures(
            metrics_dir=resolved_metrics_dir,
            figures_dir=resolved_figures_dir,
        )

    def run_full_pipeline(
        self,
        *,
        covers_manifest_path: Path,
        execute_embeddings: bool = False,
        execute_detectors: bool = False,
        include_srm: bool = True,
        skip_unimplemented: bool = False,
        quality_metrics_input: Path | None = None,
        generate_figures: bool = False,
    ) -> dict[str, Path | int]:
        """Run all non-deferred pipeline stages in sequence.

        This orchestration covers:
        - payload manifest generation
        - stego manifest generation
        - grouped split generation
        - SRM training job expansion
        - embedding stage execution/dry-run
        - detector stage execution/dry-run
        - metrics aggregation
        - optional figure generation
        """
        self.init_layout()

        resolved_covers = self._resolve_manifest_path(covers_manifest_path)
        payload_manifest = self.build_payload_manifest(
            covers_manifest_path=resolved_covers,
            write_payload_files=execute_embeddings,
        )
        stego_manifest = self.build_stego_manifest(
            covers_manifest_path=resolved_covers,
            payload_manifest_path=payload_manifest,
        )
        splits_json = self.create_grouped_splits(covers_manifest_path=resolved_covers)
        training_jobs = self.build_srm_training_jobs(splits_json_path=splits_json)
        embedding_rows = self.run_embedding_stage(
            stego_manifest_path=stego_manifest,
            execute=execute_embeddings,
        )
        predictions = self.run_detector_stage(
            stego_manifest_path=stego_manifest,
            splits_json_path=splits_json,
            execute=execute_detectors,
            include_srm=include_srm,
            skip_unimplemented=skip_unimplemented,
        )
        metrics_outputs = self.compute_metrics_from_predictions(
            predictions_path=predictions,
            quality_metrics_input=quality_metrics_input,
        )

        out: dict[str, Path | int] = {
            "payload_manifest": payload_manifest,
            "stego_manifest": stego_manifest,
            "splits_json": splits_json,
            "training_jobs": training_jobs,
            "predictions": predictions,
            "embedding_rows_processed": embedding_rows,
        }
        out.update(metrics_outputs)

        if generate_figures:
            out.update(self.generate_metrics_figures())

        return out

    def _generate_payload_bytes(self, payload_bits: int, rng: random.Random) -> bytes:
        """Generate deterministic pseudo-random payload bytes for one condition row."""
        n_bytes = payload_bits // 8
        return bytes(rng.getrandbits(8) for _ in range(n_bytes))

    def _embed_params_json(self, method: str, payload_level: str) -> str:
        """Return serialized embedding hyperparameters stored in stego manifest."""
        if method == "lsb":
            if payload_level in {"low", "medium"}:
                k = 1
                pixel_fraction = 0.25 if payload_level == "low" else 0.50
            else:
                k = 2
                pixel_fraction = 0.50
            params = {
                "method": "lsb",
                "k": k,
                "pixel_fraction": pixel_fraction,
                "prng_key": self.config.lsb_prng_key,
            }
        elif method == "dct":
            coeff_fraction = {"low": 0.10, "medium": 0.25, "high": 0.50}[payload_level]
            params = {
                "method": "dct_qim",
                "delta": self.config.dct_delta,
                "coeff_fraction": coeff_fraction,
                "zigzag_range": [10, 54],
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        return json.dumps(params, sort_keys=True)

    def _score_detector_row(
        self,
        *,
        detector: str,
        label: int,
        row: dict[str, str],
        srm_score_provider: Callable[[dict[str, str]], float] | None,
    ) -> float:
        """Score one detector on either stego (label=1) or cover (label=0) image."""
        image_path_key = "stego_path" if label == 1 else "cover_path"
        image = load_image(self._resolve_manifest_path(row[image_path_key]))

        if detector == "rs":
            return rs_analysis_score(image)
        if detector == "chi_square":
            return chi_square_score(image)
        if detector == "block_dct_shift":
            return block_dct_shift_score(image)
        if detector == "srm_ec":
            if srm_score_provider is None:
                raise NotImplementedError(
                    "SRM score provider is not wired yet. Supply srm_score_provider or disable SRM."
                )
            return float(srm_score_provider(row))
        raise ValueError(f"Unknown detector: {detector}")
