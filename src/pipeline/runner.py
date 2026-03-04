from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
import random

from src.common.contracts import ENCRYPTION_STATES, METHODS, PAYLOAD_LEVELS
from src.data.manifests import (
    CoverRecord,
    PayloadRecord,
    StegoRecord,
    TrainingJobRecord,
    read_rows_csv,
    unique_group_ids,
    write_dataclass_csv,
    write_json,
)
from src.data.images import standardize_and_save
from src.embedding.dct import embed_dct_qim
from src.embedding.encryption import encrypt_payload_aes_256_cbc
from src.embedding.lsb import embed_lsb
from src.evaluation.splits import FoldSplit, generate_grouped_5fold_splits
from src.pipeline.config import PipelineConfig


def _stable_iv(group_id: int, payload_level: str) -> bytes:
    digest = hashlib.sha256(f"{group_id}:{payload_level}".encode("utf-8")).digest()
    return digest[:16]


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.paths = config.paths

    def init_layout(self) -> None:
        self.paths.ensure_layout()

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
            standardize_and_save(
                input_path=Path(row["raw_image_path"]),
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
                    image_path=str(out_path),
                    qc_pass=row["qc_pass"].lower() == "true",
                    qc_score=float(row["qc_score"]),
                    seed=int(row["seed"]),
                )
            )

        output_path = output_manifest_path or (self.paths.manifests_dir / "covers_master.csv")
        write_dataclass_csv(output_path, records)
        return output_path

    def build_payload_manifest(
        self,
        covers_manifest_path: Path,
        output_manifest_path: Path | None = None,
        write_payload_files: bool = False,
    ) -> Path:
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
                            payload_path=str(payload_path),
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
        cover_rows = read_rows_csv(covers_manifest_path)
        payload_rows = read_rows_csv(payload_manifest_path)

        payload_index = {
            (int(row["group_id"]), row["payload_level"], row["encryption"]): row
            for row in payload_rows
        }

        records: list[StegoRecord] = []
        for cover in cover_rows:
            group_id = int(cover["group_id"])
            source = cover["source"]
            cover_path = cover["image_path"]
            for method in METHODS:
                for payload_level in PAYLOAD_LEVELS:
                    for encryption in ENCRYPTION_STATES:
                        payload_row = payload_index[(group_id, payload_level, encryption)]
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
                                payload_path=payload_row["payload_path"],
                                stego_path=str(stego_path),
                                embed_params=embed_params,
                                seed=self.config.embed_seed,
                            )
                        )

        output_path = output_manifest_path or (self.paths.manifests_dir / "stego_manifest.csv")
        write_dataclass_csv(output_path, records)
        return output_path

    def create_grouped_splits(
        self,
        covers_manifest_path: Path,
        output_path: Path | None = None,
    ) -> Path:
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
                        split_ref=str(splits_json_path),
                    )
                )

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
        from src.data.images import load_image, save_png

        for row in rows:
            cover_image = load_image(Path(row["cover_path"]))
            payload_bytes = Path(row["payload_path"]).read_bytes()
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
            save_png(stego, Path(row["stego_path"]))
        return len(rows)

    def _generate_payload_bytes(self, payload_bits: int, rng: random.Random) -> bytes:
        n_bytes = payload_bits // 8
        return bytes(rng.getrandbits(8) for _ in range(n_bytes))

    def _embed_params_json(self, method: str, payload_level: str) -> str:
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
