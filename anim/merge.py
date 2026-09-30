from __future__ import annotations

from typing import Any

from .math3d import q_rotate_vector, qidentity, qinverse, qmul, vlength, vsub
from .models import Skeleton
from .retarget import pose_world_positions


def _segment_offset(segment: dict[str, Any]) -> tuple[float, float, float]:
    offset = segment.get("root_offset", [0.0, 0.0, 0.0])
    if len(offset) != 3:
        raise ValueError("root_offset must contain three components")
    return tuple(float(value) for value in offset)  # type: ignore[return-value]


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    from math import cos, sin

    return (cos(yaw * 0.5), 0.0, sin(yaw * 0.5), 0.0)


def _place_segment_frames(
    frames: list[dict[str, Any]],
    root_name: str,
    anchor: tuple[float, float, float],
    yaw: float,
    root_bind: tuple[float, float, float, float] | None = None,
) -> None:
    yaw_q = _yaw_quaternion(yaw)
    bind = root_bind or qidentity()
    # Rigid yaw of the whole pose: global' = yaw * global, so the root local
    # rotation must be conjugated by the root bind rotation, not just premultiplied.
    local_yaw = qmul(qinverse(bind), qmul(yaw_q, bind))
    first_root = tuple(frames[0]["root_translation"])
    for frame in frames:
        delta = vsub(tuple(frame["root_translation"]), first_root)
        rotated = q_rotate_vector(yaw_q, delta)
        frame["root_translation"] = [anchor[i] + rotated[i] for i in range(3)]
        if yaw:
            rotations = frame["rotations"]
            root_rotation = tuple(rotations.get(root_name, qidentity()))
            rotations[root_name] = [float(value) for value in qmul(local_yaw, root_rotation)]


def _transform_contacts(
    contacts: list[dict[str, Any]], frame_offset: int, transform=None
) -> list[dict[str, Any]]:
    transformed = []
    for contact in contacts:
        copied = {
            **contact,
            "start": int(contact["start"]) + frame_offset,
            "end": int(contact["end"]) + frame_offset,
        }
        if transform is not None and contact.get("anchor_path"):
            copied["anchor_path"] = [list(transform(point)) for point in contact["anchor_path"]]
        transformed.append(copied)
    return transformed


def merge_segments(
    target: Skeleton,
    segments: list[dict[str, Any]],
    *,
    name: str = "merged_motion",
) -> dict[str, Any]:
    """Concatenate already-retargeted, already-grounded clips.

    Each segment keeps its own source resource, mapping version and local
    revisions.  Per-segment root translation/yaw lets clips authored in local
    coordinates be chained for turns and stair steps without re-touching
    contact poses.  Boundary gaps and foot slide across each seam are measured
    and reported, never silently hidden.
    """
    if not segments:
        raise ValueError("merge requires at least one segment")

    frames: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seams: list[dict[str, Any]] = []
    frame_rate = None
    anchor: tuple[float, float, float] | None = None

    for index, segment in enumerate(segments):
        document = segment["motion"]
        if document.get("skeleton") != target.name:
            raise ValueError(f"segment {index} does not belong to target skeleton {target.name}")
        segment_rate = float(document["frame_rate"])
        if frame_rate is None:
            frame_rate = segment_rate
        elif abs(segment_rate - frame_rate) > 1e-9:
            raise ValueError(f"segment {index} frame rate {segment_rate} != {frame_rate}")

        segment_frames = [
            {
                "root_translation": list(frame["root_translation"]),
                "rotations": {joint: list(rotation) for joint, rotation in frame["rotations"].items()},
            }
            for frame in document["frames"]
        ]

        yaw = float(segment.get("root_yaw", 0.0))
        bridge = _segment_offset(segment)
        if anchor is None:
            anchor = bridge
        first_root = tuple(document["frames"][0]["root_translation"])
        yaw_q = _yaw_quaternion(yaw)

        def place_point(point, _anchor=anchor, _yaw_q=yaw_q, _first=first_root):
            rotated = q_rotate_vector(_yaw_q, vsub(tuple(point), _first))
            return tuple(_anchor[i] + rotated[i] for i in range(3))

        _place_segment_frames(
            segment_frames, target.root, anchor, yaw,
            target.joints[target.root].bind_rotation,
        )

        start_index = len(frames)
        frames.extend(segment_frames)
        contacts.extend(_transform_contacts(document.get("contacts", []), start_index, place_point))

        last_translation = tuple(segment_frames[-1]["root_translation"])
        anchor = tuple(last_translation[i] + bridge[i] for i in range(3))

        provenance.append({
            "segment": index,
            "source_motion": document.get("source_motion"),
            "source_resource": document.get("retargeting", {}).get("source_resource", document.get("source_motion")),
            "mapping": document.get("mapping"),
            "mapping_version": document.get("mapping_version"),
            "local_revisions": document.get("local_revisions", []),
            "root_offset": list(bridge),
            "root_yaw": yaw,
            "frame_start": start_index,
            "frame_end": start_index + len(segment_frames) - 1,
        })

    for index in range(1, len(segments)):
        boundary = provenance[index]["frame_start"]
        left = frames[boundary - 1]
        right = frames[boundary]
        left_positions = pose_world_positions(target, left)
        right_positions = pose_world_positions(target, right)
        root_gap = vlength(vsub(right["root_translation"], left["root_translation"]))
        foot_slide = {}
        for joint_name in target.joints:
            if not joint_name.endswith("_foot"):
                continue
            foot_slide[joint_name] = vlength(vsub(right_positions[joint_name], left_positions[joint_name]))
        contact_pairs = {
            contact["joint"]: contact for contact in contacts
        }
        pinned_slide = {
            joint: distance
            for joint, distance in foot_slide.items()
            if any(
                contact["start"] <= boundary - 1 and contact["end"] >= boundary
                for contact in contacts if contact["joint"] == joint
            )
        }
        seams.append({
            "after_segment": index - 1,
            "frame": boundary,
            "root_gap": root_gap,
            "foot_bridge_slide": foot_slide,
            "contact_bridge_slide": pinned_slide,
            "max_contact_bridge_slide": max(pinned_slide.values(), default=0.0),
        })

    merged = {
        "name": name,
        "kind": "merged_motion",
        "skeleton": target.name,
        "target_skeleton": target.name,
        "frame_rate": frame_rate,
        "contacts": contacts,
        "frames": frames,
        "segments": provenance,
        "seams": seams,
        "merge_policy": "segments concatenated without altering grounded contact frames; "
        "per-segment root offset/yaw places clips; seam gaps are reported",
    }
    return merged


def merged_root_displacement(motion_data: dict[str, Any]) -> float:
    first = motion_data["frames"][0]["root_translation"]
    last = motion_data["frames"][-1]["root_translation"]
    return vlength(vsub(last, first))
