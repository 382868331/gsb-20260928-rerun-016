from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .io import atomic_write_json, read_json
from .pipeline import run_batch
from .sample_data import sample_batch, sample_mapping, simple_skeleton, walk_animation


def init_samples(root: Path) -> None:
    files = {
        "resources/skeletons/source_a.json": simple_skeleton("biped_source_a", 0.9, 0.16, 0.9),
        "resources/skeletons/target_b.json": simple_skeleton("biped_target_b", 1.25, 0.20, 1.25),
        "resources/animations/source_walk.json": walk_animation("biped_source_a"),
        "mappings/biped_a_to_b.json": sample_mapping(),
        "batches/sample_walk_transfer.json": sample_batch(),
    }
    for relative, payload in files.items():
        atomic_write_json(root / relative, payload)


def execute_batch(root: Path, batch_path: Path) -> Path:
    batch = read_json(batch_path)
    results = run_batch(root, batch)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = root / "batches" / "runs" / f"{batch['name']}-{timestamp}.json"
    atomic_write_json(
        report_path,
        {
            "batch": str(batch_path),
            "name": batch["name"],
            "created_at": timestamp,
            "passed": all(result.passed for result in results),
            "results": [
                {"output": result.output, "passed": result.passed, "errors": result.errors}
                for result in results
            ],
        },
        overwrite=True,
    )
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Animation resource processing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init-samples")
    init_parser.add_argument("--root", type=Path, default=Path("data"))
    run_parser = subparsers.add_parser("run-batch")
    run_parser.add_argument("--root", type=Path, default=Path("data"))
    run_parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "init-samples":
        init_samples(args.root)
        print(f"sample data written under {args.root}")
        return 0
    if args.command == "run-batch":
        report = execute_batch(args.root, args.batch)
        print(f"report: {report}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
