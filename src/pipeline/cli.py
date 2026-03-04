from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline scaffold utilities for project-proposal."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current working directory).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-layout", help="Create expected data/results directory layout.")

    p_std = sub.add_parser(
        "standardize-covers",
        help="Read raw cover index CSV and write canonical covers_master.csv + standardized PNG covers.",
    )
    p_std.add_argument(
        "--input-index",
        type=Path,
        required=True,
        help="CSV with raw_image_path and cover metadata.",
    )

    p_payload = sub.add_parser("build-payload-manifest", help="Write payload manifest.")
    p_payload.add_argument(
        "--covers-manifest",
        type=Path,
        required=True,
        help="Path to covers_master.csv",
    )
    p_payload.add_argument(
        "--write-files",
        action="store_true",
        help="Write payload binary files. Encrypted branch calls placeholder AES implementation.",
    )

    p_stego = sub.add_parser("build-stego-manifest", help="Write stego manifest.")
    p_stego.add_argument("--covers-manifest", type=Path, required=True)
    p_stego.add_argument("--payload-manifest", type=Path, required=True)

    p_split = sub.add_parser("create-splits", help="Create grouped 5-fold split JSON.")
    p_split.add_argument("--covers-manifest", type=Path, required=True)

    p_jobs = sub.add_parser(
        "build-training-jobs",
        help="Write SRM per-method fold job manifest from split JSON.",
    )
    p_jobs.add_argument("--splits-json", type=Path, required=True)

    p_embed = sub.add_parser(
        "run-embedding-stage",
        help="Run embedding stage from stego manifest (placeholder embedding functions).",
    )
    p_embed.add_argument("--stego-manifest", type=Path, required=True)
    p_embed.add_argument(
        "--execute",
        action="store_true",
        help="Actually invoke embedding placeholders; default is dry-run count only.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    config = PipelineConfig.from_project_root(project_root)
    runner = PipelineRunner(config)

    if args.command == "init-layout":
        runner.init_layout()
        print("Layout initialized.")
    elif args.command == "standardize-covers":
        out = runner.standardize_covers_from_index(
            input_index_csv=_resolve_path(args.input_index, project_root),
        )
        print(f"Covers manifest: {out}")
    elif args.command == "build-payload-manifest":
        out = runner.build_payload_manifest(
            covers_manifest_path=_resolve_path(args.covers_manifest, project_root),
            write_payload_files=args.write_files,
        )
        print(f"Payload manifest: {out}")
    elif args.command == "build-stego-manifest":
        out = runner.build_stego_manifest(
            covers_manifest_path=_resolve_path(args.covers_manifest, project_root),
            payload_manifest_path=_resolve_path(args.payload_manifest, project_root),
        )
        print(f"Stego manifest: {out}")
    elif args.command == "create-splits":
        out = runner.create_grouped_splits(
            covers_manifest_path=_resolve_path(args.covers_manifest, project_root)
        )
        print(f"Splits JSON: {out}")
    elif args.command == "build-training-jobs":
        out = runner.build_srm_training_jobs(
            splits_json_path=_resolve_path(args.splits_json, project_root)
        )
        print(f"Training jobs CSV: {out}")
    elif args.command == "run-embedding-stage":
        n = runner.run_embedding_stage(
            stego_manifest_path=_resolve_path(args.stego_manifest, project_root),
            execute=args.execute,
        )
        print(f"Embedding rows processed: {n}")
    else:
        raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
