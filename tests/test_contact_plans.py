import json
import tempfile
import unittest
from pathlib import Path

from anim.contacts import ContactInterval, apply_contact_constraints, parse_contacts
from anim.merge import merge_segments
from anim.models import Motion, load_mapping, load_motion, load_skeleton
from anim.plans import compare_contact_plans, evaluate_contact_plan, save_plan_document
from anim.retarget import pose_world_positions, retarget_motion
from anim.validation import (
    check_contact_sliding,
    expected_contact_position,
    validate_grounded,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


def setup(motion_name="railing_step_source", target_name="simple_target"):
    source = load_skeleton(RESOURCES / "skeletons/simple_source.json")
    target = load_skeleton(RESOURCES / "skeletons" / f"{target_name}.json")
    motion = load_motion(RESOURCES / "motions" / f"{motion_name}.json")
    mapping = load_mapping(RESOURCES / "mappings" / f"simple_source_to_{target_name}.json")
    document = retarget_motion(source, target, motion, mapping).document
    return source, target, motion, mapping, document


def railing_plan(overrides=None):
    _, _, _, _, document = setup()
    scale = document["root_scale"]
    k1 = [0.10 / scale, 2.20 / scale, 0.32 / scale]
    k4 = [0.10 / scale, 2.20 / scale, 0.42 / scale]
    contacts = [
        {"joint": "r_foot", "start": 1, "end": 5, "ground_y": 0.15,
         "priority": 2, "blend_in": 1, "blend_out": 1},
        {"joint": "l_foot", "start": 0, "end": 2, "ground_y": 0.1119,
         "priority": 2, "blend_out": 1},
        {"joint": "l_hand", "start": 1, "end": 4,
         "priority": 2, "blend_in": 1, "blend_out": 1,
         "target_keyframes": [
             {"frame": 1, "position": k1},
             {"frame": 4, "position": k4},
         ]},
    ]
    if overrides:
        by_joint = {item["joint"]: item for item in contacts}
        for joint, values in overrides.items():
            by_joint[joint].update(values)
    return contacts


class OverlappingContactTests(unittest.TestCase):
    def test_two_feet_and_hand_contact_overlap(self):
        _, target, _, _, document = setup()
        report = apply_contact_constraints(
            target, document, ground_scale=document["root_scale"],
            contact_plan=railing_plan(),
        )
        self.assertEqual(
            {item["joint"] for item in report["contacts"]},
            {"r_foot", "l_foot", "l_hand"},
        )
        self.assertTrue(all(item["status"] == "satisfied" for item in report["contacts"]))
        self.assertLess(report["max_contact_error"], 1e-6)
        motion = Motion(document)
        self.assertEqual(check_contact_sliding(target, motion, tolerance=0.005), [])

    def test_overlapping_contacts_work_for_tall_skeleton(self):
        source, target, motion, mapping, _ = setup(target_name="simple_tall_target")
        result = evaluate_contact_plan(
            source, target, motion, mapping,
            {"name": "tall", "contacts": railing_plan()},
        )
        self.assertNotEqual(result["metrics"]["root_scale"], 1.0)
        self.assertEqual(result["metrics"]["satisfied"], 3)
        self.assertTrue(result["validation"]["passed"])

    def test_frames_outside_contacts_keep_original_motion(self):
        _, target, _, _, untouched = setup()
        _, target_b, _, _, document = setup()
        apply_contact_constraints(
            target_b, document, ground_scale=document["root_scale"],
            contact_plan=railing_plan(),
        )
        # frame 0 has only l_foot contact; the hand is free and must keep the
        # retargeted pose rather than be pulled toward the future railing.
        free_before = pose_world_positions(target, untouched["frames"][0])["l_hand"]
        free_after = pose_world_positions(target_b, document["frames"][0])["l_hand"]
        distance = sum((a - b) ** 2 for a, b in zip(free_before, free_after)) ** 0.5
        self.assertLess(distance, 1e-8)

    def test_entry_and_exit_blends_are_smooth(self):
        _, target, _, _, document = setup()
        plan = railing_plan()
        apply_contact_constraints(
            target, document, ground_scale=document["root_scale"], contact_plan=plan
        )
        # full-weight hand frames 2 and 3 must sit exactly on the moving rail
        hand_contacts = [c for c in document["contacts"] if c["joint"] == "l_hand"][0]
        for frame_index in (2, 3):
            expected = expected_contact_position(hand_contacts, frame_index)
            actual = pose_world_positions(target, document["frames"][frame_index])["l_hand"]
            self.assertLess(
                sum((a - b) ** 2 for a, b in zip(expected, actual)) ** 0.5,
                1e-6,
            )
        # blend frame 1 lies between the free pose and the rail (no teleport)
        free = pose_world_positions(target, setup()[4]["frames"][1])["l_hand"]
        first_target = hand_contacts["target_keyframes"][0]["position"]
        blended = pose_world_positions(target, document["frames"][1])["l_hand"]
        free_distance = sum((a - b) ** 2 for a, b in zip(free, blended)) ** 0.5
        target_distance = sum((a - b) ** 2 for a, b in zip(first_target, blended)) ** 0.5
        self.assertGreater(free_distance, 1e-4)
        self.assertGreater(target_distance, 1e-4)
        self.assertLess(free_distance, sum((a - b) ** 2 for a, b in zip(free, first_target)) ** 0.5)

    def test_invalid_blend_window_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_contacts(
                [{"joint": "l_foot", "start": 0, "end": 1, "blend_in": 2}],
                4, {"l_foot"},
            )


class MovingSupportTests(unittest.TestCase):
    def test_feet_follow_moving_platform_keyframes(self):
        source = load_skeleton(RESOURCES / "skeletons/simple_source.json")
        target = load_skeleton(RESOURCES / "skeletons/simple_target.json")
        motion = load_motion(RESOURCES / "motions/moving_platform_source.json")
        mapping = load_mapping(RESOURCES / "mappings/simple_source_to_simple_target.json")
        document = retarget_motion(source, target, motion, mapping).document
        scale = document["root_scale"]
        lateral = 0.1
        left_keys = [
            {"frame": frame,
             "position": [0.0, 0.1119 + 0.02 * frame, 0.10 * frame - lateral]}
            for frame in range(4)
        ]
        right_keys = [
            {"frame": frame,
             "position": [0.0, 0.1119 + 0.02 * frame, 0.10 * frame + lateral]}
            for frame in range(4)
        ]
        plan = [
            {"joint": "l_foot", "start": 0, "end": 3, "priority": 1,
             "blend_out": 1, "target_keyframes": left_keys},
            {"joint": "r_foot", "start": 0, "end": 3, "priority": 1,
             "blend_out": 1, "target_keyframes": right_keys},
        ]
        report = apply_contact_constraints(target, document, ground_scale=scale, contact_plan=plan)
        self.assertEqual(len(report["unmet"]), 0)
        validation = validate_grounded(target, Motion(document), report, slide_tolerance=0.005)
        self.assertTrue(validation["passed"], validation["failures"])
        first = pose_world_positions(target, document["frames"][0])["l_foot"]
        last = pose_world_positions(target, document["frames"][2])["l_foot"]
        self.assertAlmostEqual(last[2] - first[2], 0.20 * scale, places=5)
        self.assertAlmostEqual(last[1] - first[1], 0.04 * scale, places=5)


class PriorityConcessionTests(unittest.TestCase):
    def far_rail_plan(self, hand_priority, feet_priority):
        _, _, _, _, document = setup()
        scale = document["root_scale"]
        return [
            {"joint": "r_foot", "start": 1, "end": 5, "ground_y": 0.15,
             "priority": feet_priority},
            {"joint": "l_foot", "start": 0, "end": 2, "ground_y": 0.1119,
             "priority": feet_priority},
            {"joint": "l_hand", "start": 1, "end": 4, "priority": hand_priority,
             "target_keyframes": [
                 {"frame": 1, "position": [1.5 / scale, 3.0 / scale, 1.5 / scale]},
                 {"frame": 4, "position": [1.6 / scale, 3.1 / scale, 1.6 / scale]},
             ]},
        ]

    def test_lower_priority_hand_concedes_and_error_is_reported(self):
        source, target, motion, mapping, _ = setup()
        result = evaluate_contact_plan(
            source, target, motion, mapping,
            {"name": "feet_first", "contacts": self.far_rail_plan(9, 1)},
        )
        metrics = result["metrics"]
        self.assertEqual(metrics["satisfied"], 2)
        self.assertEqual(metrics["conceded"], 1)
        hand = next(item for item in result["grounding_report"]["contacts"]
                    if item["joint"] == "l_hand")
        self.assertEqual(hand["status"], "conceded")
        self.assertTrue(set(hand["conceded_to"]) & {"l_foot", "r_foot"})
        self.assertGreater(hand["max_error"], 0.5)
        # the concession is reported, not hidden as success, and it is not a
        # slide failure because the contact explicitly yielded
        self.assertTrue(result["validation"]["passed"])
        self.assertEqual(result["validation"]["failure_count"], 0)
        self.assertGreaterEqual(result["validation"]["concession_count"], 1)

    def test_swapping_priority_changes_who_yields(self):
        source, target, motion, mapping, _ = setup()
        feet_first = evaluate_contact_plan(
            source, target, motion, mapping,
            {"name": "feet_first", "contacts": self.far_rail_plan(9, 1)},
        )
        hand_first = evaluate_contact_plan(
            source, target, motion, mapping,
            {"name": "hand_first", "contacts": self.far_rail_plan(1, 9)},
        )
        self.assertEqual(
            [i for i in feet_first["grounding_report"]["contacts"] if i["status"] == "conceded"][0]["joint"],
            "l_hand",
        )
        conceded_joints = {
            i["joint"] for i in hand_first["grounding_report"]["contacts"]
            if i["status"] == "conceded"
        }
        self.assertEqual(conceded_joints, {"l_foot", "r_foot"})


class PlanComparisonTests(unittest.TestCase):
    def test_plans_are_ranked_and_selection_can_be_saved(self):
        source, target, motion, mapping, _ = setup()
        plans_document = json.loads(
            (ROOT / "data" / "contact_plans" / "railing_step_plans.json").read_text()
        )
        result = compare_contact_plans(source, target, motion, mapping, plans_document["plans"])
        names = {item["name"]: item["rank"] for item in result["comparison"]}
        self.assertEqual(names["balanced"], 1)
        self.assertEqual(result["selected"], "balanced")
        selected = next(e for e in result["evaluations"] if e["name"] == "balanced")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.json"
            save_plan_document(selected, path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["grounding"]["plan"], "balanced")
            reloaded = Motion(saved)
            self.assertEqual(check_contact_sliding(target, reloaded, tolerance=0.005), [])


class SavedRecheckTests(unittest.TestCase):
    def test_saved_motion_rechecks_turn_stairs_and_merge(self):
        from anim.pipeline import run_batch

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_batch(
                ROOT / "data" / "resources",
                ROOT / "data" / "batches" / "multi_segment_batch.json",
                tmp,
            )
            self.assertEqual(summary["failed"], 0, summary)
            for job in summary["jobs"]:
                verification = job["validation"].get("saved_verification")
                self.assertIsNotNone(verification, job["job"])
                self.assertTrue(verification["passed"], job["job"])

    def test_merged_motion_transforms_moving_keyframes_with_yaw(self):
        _, target, _, _, document = setup()
        apply_contact_constraints(
            target, document, ground_scale=document["root_scale"],
            contact_plan=railing_plan(),
        )
        merged = merge_segments(
            target,
            [
                {"motion": document, "root_yaw": 0.3},
                {"motion": document, "root_yaw": -0.3, "root_offset": [0.0, 0.05, 0.0]},
            ],
        )
        motion = Motion(merged)
        self.assertEqual(check_contact_sliding(target, motion, tolerance=0.005), [])
        hand_contacts = [c for c in merged["contacts"] if c["joint"] == "l_hand"]
        self.assertTrue(all(c["moving"] for c in hand_contacts))


if __name__ == "__main__":
    unittest.main()
