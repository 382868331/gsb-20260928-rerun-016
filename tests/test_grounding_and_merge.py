import json
import tempfile
import unittest
from pathlib import Path

from anim.contacts import apply_foot_contacts, parse_contacts
from anim.merge import merge_segments, merged_root_displacement
from anim.models import Motion, load_mapping, load_motion, load_skeleton
from anim.retarget import pose_world_positions, retarget_motion
from anim.validation import check_contact_sliding, validate_grounded, validate_retarget


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


def build(skeleton_name, motion_name, mapping_name):
    source = load_skeleton(RESOURCES / "skeletons" / "simple_source.json")
    target = load_skeleton(RESOURCES / "skeletons" / f"{skeleton_name}.json")
    motion = load_motion(RESOURCES / "motions" / f"{motion_name}.json")
    mapping = load_mapping(RESOURCES / "mappings" / f"{mapping_name}.json")
    document = retarget_motion(source, target, motion, mapping).document
    report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
    return source, target, motion, mapping, document, report


class GroundingTests(unittest.TestCase):
    def test_contact_feet_do_not_slide_and_swing_feet_are_free(self):
        _, target, _, _, document, report = build(
            "simple_target", "walk_ground_source", "simple_source_to_simple_target"
        )
        motion = Motion(document)
        self.assertEqual(check_contact_sliding(target, motion, tolerance=0.005), [])
        grounded = validate_grounded(target, motion, report)
        self.assertTrue(grounded["passed"], grounded)

        first_lift = pose_world_positions(target, document["frames"][1])["l_foot"]
        first_press = pose_world_positions(target, document["frames"][0])["l_foot"]
        self.assertGreater(
            sum((a - b) ** 2 for a, b in zip(first_lift, first_press)) ** 0.5,
            0.05,
        )

    def test_grounding_pins_different_leg_length_and_keeps_mapping_version(self):
        _, target, _, mapping, document, report = build(
            "simple_tall_target", "turn_source", "simple_source_to_simple_tall_target"
        )
        self.assertEqual(document["mapping_version"], mapping.version)
        self.assertNotEqual(document["root_scale"], 1.0)
        slides = check_contact_sliding(target, Motion(document), tolerance=0.005)
        self.assertEqual(slides, [])
        self.assertLess(report["max_contact_error"], 0.005)

    def test_stair_contacts_anchor_to_step_levels(self):
        _, target, _, _, document, report = build(
            "simple_target", "stairs_source", "simple_source_to_simple_target"
        )
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])
        levels = {
            (item["joint"], item["start"]): item["ground_y"] for item in report["contacts"]
        }
        self.assertAlmostEqual(levels[("r_foot", 0)], 0.1119 * document["root_scale"], places=4)
        self.assertAlmostEqual(levels[("l_foot", 2)], 0.1919 * document["root_scale"], places=4)
        pinned = pose_world_positions(target, document["frames"][3])["l_foot"]
        self.assertAlmostEqual(pinned[1], 0.1919 * document["root_scale"], places=3)

    def test_limit_conflict_is_reported_not_forced(self):
        _, target, _, _, document, _ = build(
            "simple_target", "walk_ground_source", "simple_source_to_simple_target"
        )
        strict_target = json.loads((RESOURCES / "skeletons" / "simple_target.json").read_text())
        for joint_name in ("l_knee", "r_knee"):
            strict_target["joints"][joint_name]["rotation_limits"]["x"] = [-0.02, 0.0]
        strict = type(target)(strict_target)
        report = apply_foot_contacts(strict, document, ground_scale=document["root_scale"])
        conflict_joints = {
            conflict["joint"]
            for item in report["unmet"]
            for conflict in item["limit_conflicts"]
        }
        self.assertTrue(conflict_joints & {"l_knee", "r_knee"})
        for item in report["unmet"]:
            self.assertIn("residual", item)
            self.assertIn("anchor", item)

    def test_local_revisions_are_preserved_on_output(self):
        _, _, source_motion, _, document, _ = build(
            "simple_target", "walk_ground_source", "simple_source_to_simple_target"
        )
        self.assertEqual(
            document["local_revisions"][0]["revision"],
            source_motion.local_revisions[0]["revision"],
        )

    def test_invalid_contact_range_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_contacts([{"joint": "l_foot", "start": 0, "end": 9}], 4, {"l_foot"})


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.target = load_skeleton(RESOURCES / "skeletons" / "simple_target.json")
        segments = []
        for name in ("walk_ground_source", "turn_source", "stairs_source"):
            _, _, _, _, document, _ = build(
                "simple_target", name, "simple_source_to_simple_target"
            )
            segments.append(document)
        self.segments = segments

    def test_merge_keeps_provenance_and_continuous_roots(self):
        merged = merge_segments(
            self.target,
            [
                {"motion": self.segments[0]},
                {"motion": self.segments[1], "root_yaw": 0.2},
                {"motion": self.segments[2], "root_offset": [0.0, 0.02, 0.0]},
            ],
            name="walk_turn_stairs",
        )
        self.assertEqual(len(merged["frames"]), 12)
        self.assertEqual(len(merged["segments"]), 3)
        versions = {item["mapping_version"] for item in merged["segments"]}
        self.assertIn("2026.09-v1", versions)
        self.assertTrue(all(item["local_revisions"] for item in merged["segments"]))
        for seam in merged["seams"]:
            self.assertLess(seam["root_gap"], 1e-6)
        self.assertGreater(merged_root_displacement(merged), 1.0)
        motion = Motion(merged)
        self.assertEqual(check_contact_sliding(self.target, motion, tolerance=0.005), [])

    def test_merge_rejects_frame_rate_mismatch(self):
        changed = json.loads(json.dumps(self.segments[1]))
        changed["frame_rate"] = 12.0
        with self.assertRaises(ValueError):
            merge_segments(
                self.target,
                [{"motion": self.segments[0]}, {"motion": changed}],
            )


if __name__ == "__main__":
    unittest.main()
