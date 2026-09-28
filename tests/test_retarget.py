from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anim_pipeline.fk import forward_kinematics
from anim_pipeline.io import load_animation, load_mapping, load_skeleton
from anim_pipeline.pipeline import run_batch
from anim_pipeline.retarget import retarget
from anim_pipeline.validate import validate_animation
from anim_pipeline.sample_data import sample_mapping, simple_skeleton, walk_animation
from anim_pipeline.models import Mapping


def test_retarget_scales_root_and_feet_for_different_bone_lengths(tmp_path: Path):
    source = load_skeleton_from_dict(simple_skeleton("biped_source_a", 0.9, 0.16, 0.9))
    target = load_skeleton_from_dict(simple_skeleton("biped_target_b", 1.25, 0.2, 1.25))
    mapping = Mapping.from_dict(sample_mapping())
    animation = load_animation_from_dict(walk_animation("biped_source_a"))

    result = retarget(source, target, animation, mapping)
    assert result.scale == pytest.approx(1.25 / 0.9)
    report = validate_animation(
        target,
        result.animation,
        source_animation=animation,
        source_skeleton=source,
        scale=result.scale,
    )
    assert report.passed, report.errors
    root_gain = result.animation.root_translations[-1] - result.animation.root_translations[0]
    assert root_gain[0] == pytest.approx(0.42 * result.scale)
    assert result.unmapped_source == ["source_ponytail"]
    assert result.unmapped_target == ["target_tail"]


def test_required_missing_joint_rejected():
    source = load_skeleton_from_dict(simple_skeleton("biped_source_a", 0.9, 0.16, 0.9))
    target_data = simple_skeleton("biped_target_b", 1.25, 0.2, 1.25)
    removed = {"left_foot", "left_toe"}
    target_data["joints"] = [j for j in target_data["joints"] if j["name"] not in removed]
    target = load_skeleton_from_dict(target_data)
    mapping = Mapping.from_dict(sample_mapping())
    animation = load_animation_from_dict(walk_animation("biped_source_a"))

    with pytest.raises(ValueError, match="incomplete joint mapping"):
        retarget(source, target, animation, mapping)


def test_failed_batch_does_not_create_output_and_existing_output_is_protected(tmp_path: Path):
    resources = tmp_path / "resources"
    mappings = tmp_path / "mappings"
    batches = tmp_path / "batches"
    (resources / "skeletons").mkdir(parents=True)
    (resources / "animations").mkdir()
    mappings.mkdir()
    batches.mkdir()
    write(resources / "skeletons/source.json", simple_skeleton("biped_source_a", 0.9, 0.16, 0.9))
    write(resources / "skeletons/target.json", simple_skeleton("biped_target_b", 1.25, 0.2, 1.25))
    write(mappings / "map.json", sample_mapping())
    write(resources / "animations/walk.json", walk_animation("biped_source_a"))
    output = resources / "animations/out.json"
    batch = {
        "name": "test",
        "jobs": [
            {
                "source_skeleton": "resources/skeletons/source.json",
                "target_skeleton": "resources/skeletons/nonexistent.json",
                "mapping": "mappings/map.json",
                "source_animation": "resources/animations/walk.json",
                "output_animation": "resources/animations/out.json",
            }
        ],
    }
    first = run_batch(tmp_path, batch)
    assert not first[0].passed
    assert not output.exists()

    output.write_text("{}", encoding="utf-8")
    second = run_batch(tmp_path, batch)
    assert not second[0].passed
    assert output.read_text(encoding="utf-8") == "{}"


def load_skeleton_from_dict(data):
    from anim_pipeline.models import Skeleton

    return Skeleton.from_dict(data)


def load_animation_from_dict(data):
    from anim_pipeline.models import Animation

    return Animation.from_dict(data)


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
