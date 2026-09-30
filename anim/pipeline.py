from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .models import Mapping, Motion, Skeleton, atomic_write_json, load_batch, load_json, load_mapping, load_motion, load_skeleton
from .contact import solve_contacts
from .merge import merge_clips
from .merge_validation import validate_merged_motion
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
            if job.get("kind") == "merge_clips":
                target = load_skeleton(base / job["target_skeleton"])
                clips = []
                for clip in job["clips"]:
                    clips.append({
                        "source_skeleton": load_skeleton(base / clip["source_skeleton"]),
                        "source_motion": load_motion(base / clip["source_motion"]),
                        "mapping": load_mapping(base / clip["mapping"]),
                        "source_path": clip["source_motion"],
                        "revisions": clip.get("revisions", []),
                    })
                merged_document = merge_clips(clips, name=job.get("output_name", name), target=target)
                validation = validate_merged_motion(target, merged_document)
                if not validation["passed"]:
                    raise ValueError(f"merge validation failed: {validation['failure_count']} issue(s)")
                motion_output = output / job["output_motion"]
                atomic_write_json(motion_output, merged_document, overwrite=overwrite)
                record = {
                    "source_motion_resource": [clip["source_motion"] for clip in job["clips"]],
                    "converted_motion_resource": job["output_motion"],
                    "validation": validation,
                    "retargeting_metadata": merged_document["merge_metadata"],
                }
                results.append(JobResult(name, "success", str(motion_output), validation=record).__dict__)
                continue
            source = load_skeleton(base / job["source_skeleton"])
            target = load_skeleton(base / job["target_skeleton"])
            source_motion = load_motion(base / job["source_motion"])
            mapping = load_mapping(base / job["mapping"])
            retargeted = retarget_motion(source, target, source_motion, mapping)
            target_motion_data = retargeted.document
            foot_pairs = {
                source_name: target_name
                for source_name, target_name in mapping.required_pairs().items()
                if source_name.endswith("_foot")
            }
            if source_motion.contacts and foot_pairs:
                grounded = solve_contacts(
                    target, Motion(target_motion_data), source_motion, foot_pairs
                )
                target_motion_data["frames"] = grounded["frames"]
                target_motion_data["contact_solve"] = {
                    "anchors": grounded["anchors"],
                    "slides": grounded["slides"],
                    "unmet_contacts": grounded["unmet"],
                    "policy": "contact frames pinned; free frames untouched; limit conflicts report residuals",
                }
            target_motion = Motion(target_motion_data)
            has_contacts = bool(source_motion.contacts) and "contact_solve" in target_motion_data
            validation = validate_retarget(source, target, source_motion, target_motion, mapping, skip_foot_position=has_contacts)
            if has_contacts:
                contact_validation = validate_merged_motion(target, {
                    **target_motion_data,
                    "provenance": [{"unmet_contacts": target_motion_data["contact_solve"]["unmet_contacts"]}],
                })
                validation["passed"] = validation["passed"] and contact_validation["passed"]
                validation["contact"] = {
                    key: value
                    for key, value in contact_validation.items()
                    if key in ("slides", "unmet_contacts", "failures")
                }
                validation["failure_count"] += contact_validation["failure_count"]
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
