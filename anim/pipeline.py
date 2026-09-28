from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .models import Mapping, Motion, Skeleton, atomic_write_json, load_batch, load_json, load_mapping, load_motion, load_skeleton
from .retarget import retarget_motion
from .validation import validate_retarget


@dataclass
class JobResult:
    job: str
    status: str
    output: str | None = None
    error: str | None = None
    validation: dict[str, Any] | None = None


def run_batch(
    base_dir: str | Path,
    batch_path: str | Path,
    output_dir: str | Path,
    *, overwrite: bool = False,
) -> dict[str, Any]:
    base = Path(base_dir)
    output = Path(output_dir)
    batch = load_batch(batch_path)
    results = []

    for job in batch.jobs:
        name = job["name"]
        try:
            source = load_skeleton(base / job["source_skeleton"])
            target = load_skeleton(base / job["target_skeleton"])
            source_motion = load_motion(base / job["source_motion"])
            mapping = load_mapping(base / job["mapping"])
            retargeted = retarget_motion(source, target, source_motion, mapping)
            target_motion_data = retargeted.document
            target_motion = Motion(target_motion_data)
            validation = validate_retarget(source, target, source_motion, target_motion, mapping)
            if not validation["passed"]:
                raise ValueError(f"validation failed: {validation['failure_count']} issue(s)")

            motion_output = output / job["output_motion"]
            atomic_write_json(motion_output, target_motion_data, overwrite=overwrite)
            record = {
                "source_motion_resource": job["source_motion"],
                "converted_motion_resource": job["output_motion"],
                "validation": validation,
                "retargeting_metadata": target_motion_data["retargeting"],
            }
            results.append(JobResult(name, "success", str(motion_output), validation=record).__dict__)
        except Exception as exc:
            results.append(JobResult(name, "failed", error=str(exc)).__dict__)

    summary = {
        "batch": batch.name,
        "total": len(results),
        "succeeded": sum(1 for item in results if item["status"] == "success"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "jobs": results,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    atomic_write_json(output / "records" / f"{batch.name}_{timestamp}.json", summary, overwrite=overwrite)
    return summary
