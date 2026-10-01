from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .models import Mapping, Motion, atomic_write_json, load_batch, load_mapping, load_motion, load_skeleton
from .contacts import apply_foot_contacts
from .handoffs import attach_handoff_audit
from .merge import merge_segments, merged_root_displacement
from .retarget import retarget_motion
from .schemes import compare_schemes
from .validation import validate_grounded, validate_retarget


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
            if job.get("type") == "compare":
                source = load_skeleton(base / job["source_skeleton"])
                target = load_skeleton(base / job["target_skeleton"])
                source_motion = load_motion(base / job["source_motion"])
                mapping = load_mapping(base / job["mapping"])
                retargeted = retarget_motion(source, target, source_motion, mapping)
                comparison = compare_schemes(
                    target,
                    retargeted.document,
                    job["schemes"],
                    ground_scale=retargeted.document["root_scale"],
                )
                report_output = output / job["output_report"]
                saved_report = {
                    key: value
                    for key, value in comparison.items()
                    if key not in {"documents", "grounding_reports", "handoff_audits"}
                }
                saved_report["handoff_audits"] = comparison["handoff_audits"]
                atomic_write_json(report_output, saved_report, overwrite=overwrite)
                record = {
                    "source_motion_resource": job["source_motion"],
                    "comparison_report": job["output_report"],
                    "schemes": saved_report["schemes"],
                    "recommended": comparison["recommended"],
                    "mapping_version": retargeted.document["mapping_version"],
                }
                selected = job.get("selected_scheme")
                if selected is not None:
                    if selected not in comparison["documents"]:
                        raise ValueError(f"unknown selected scheme: {selected}")
                    selected_document = comparison["documents"][selected]
                    grounding = comparison["grounding_reports"][selected]
                    grounding_validation = validate_grounded(
                        target, Motion(selected_document), grounding, slide_tolerance=0.005
                    )
                    if not grounding_validation["passed"]:
                        raise ValueError(
                            f"selected scheme {selected} failed grounding: "
                            f"{grounding_validation['failure_count']} contact issue(s)"
                        )
                    motion_output = output / job["output_motion"]
                    atomic_write_json(motion_output, selected_document, overwrite=overwrite)
                    record["selected_scheme"] = selected
                    record["converted_motion_resource"] = job["output_motion"]
                    record["grounding"] = grounding
                results.append(JobResult(name, "success", str(report_output), validation=record).__dict__)
                continue

            if job.get("type", "retarget") == "merge":
                target = load_skeleton(base / job["target_skeleton"])
                segments = []
                for segment in job["segments"]:
                    segment_motion_path = output / segment["motion"]
                    segment_document = load_motion(segment_motion_path)
                    segment_data = json.loads(segment_motion_path.read_text(encoding="utf-8"))
                    segments.append({
                        "motion": segment_data,
                        "root_offset": segment.get("root_offset", [0.0, 0.0, 0.0]),
                        "root_yaw": segment.get("root_yaw", 0.0),
                    })
                merged_data = merge_segments(
                    target, segments, name=job.get("merged_name", f"{name}_merged")
                )
                merged_motion = Motion(merged_data)
                grounding_report = {"contacts": merged_data.get("contacts", []), "unmet": [], "max_contact_error": 0.0}
                validation = validate_grounded(
                    target, merged_motion, grounding_report, slide_tolerance=0.005
                )
                if not validation["passed"]:
                    raise ValueError(f"merge validation failed: {validation['failure_count']} issue(s)")
                merge_output = output / job["output_motion"]
                atomic_write_json(merge_output, merged_data, overwrite=overwrite)
                record = {
                    "source_segments": [
                        segment["motion"] for segment in job["segments"]
                    ],
                    "converted_motion_resource": job["output_motion"],
                    "validation": validation,
                    "root_displacement": merged_root_displacement(merged_data),
                    "seams": merged_data["seams"],
                    "segments": merged_data["segments"],
                }
                results.append(JobResult(name, "success", str(merge_output), validation=record).__dict__)
                continue

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

            grounding_report = apply_foot_contacts(
                target,
                target_motion_data,
                ground_scale=target_motion_data["root_scale"],
                manual_pins=target_motion_data.get("manual_pins"),
                freeze_before=(
                    max((int(pin["freeze_before"]) for pin in target_motion_data.get("manual_pins", [])), default=None)
                ),
            )
            attach_handoff_audit(target, target_motion_data, grounding_report)
            grounded_motion = Motion(target_motion_data)
            grounding_validation = validate_grounded(
                target, grounded_motion, grounding_report, slide_tolerance=0.005
            )
            if not grounding_validation["passed"]:
                raise ValueError(
                    f"grounding failed: {grounding_validation['failure_count']} contact issue(s)"
                )
            validation["grounding"] = grounding_validation

            motion_output = output / job["output_motion"]
            atomic_write_json(motion_output, target_motion_data, overwrite=overwrite)
            record = {
                "source_motion_resource": job["source_motion"],
                "converted_motion_resource": job["output_motion"],
                "validation": validation,
                "grounding": grounding_report,
                "mapping_version": target_motion_data["mapping_version"],
                "local_revisions": target_motion_data.get("local_revisions", []),
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
