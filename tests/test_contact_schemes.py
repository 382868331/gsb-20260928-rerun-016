import json
import tempfile
import unittest
from pathlib import Path

from anim.contacts import apply_foot_contacts, parse_contacts
from anim.models import Motion, load_mapping, load_motion, load_skeleton
from anim.pipeline import run_batch
from anim.retarget import pose_world_positions, retarget_motion
from anim.schemes import compare_schemes
from anim.handoffs import build_handoff_audit
from anim.validation import check_contact_sliding, validate_grounded


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data" / "resources"


def build(skeleton_name, motion_name, mapping_name):
    source = load_skeleton(RESOURCES / "skeletons" / "simple_source.json")
    target = load_skeleton(RESOURCES / "skeletons" / f"{skeleton_name}.json")
    motion = load_motion(RESOURCES / "motions" / f"{motion_name}.json")
    mapping = load_mapping(RESOURCES / "mappings" / f"{mapping_name}.json")
    document = retarget_motion(source, target, motion, mapping).document
    return source, target, motion, mapping, document


def build_railing(skeleton_name="simple_target", mapping_name="simple_source_to_simple_target"):
    return build(skeleton_name, "railing_source", mapping_name)


class OverlappingContactTests(unittest.TestCase):
    def test_overlapping_feet_and_hand_all_satisfied(self):
        _, target, _, _, document = build_railing()
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        by_name = {contact["contact"]: contact for contact in report["contacts"]}
        self.assertEqual(report["unmet"], [])
        self.assertTrue(by_name["step_edge_foot"]["met"])
        self.assertTrue(by_name["step_top_foot"]["met"])
        self.assertTrue(by_name["escalator_rail"]["met"])
        # frame 2 sits inside all three intervals: overlap is really solved
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])

    def test_hand_rides_moving_rail_and_foot_rides_moving_step(self):
        _, target, _, _, document = build_railing()
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        moving = {c["contact"] for c in report["contacts"] if c["moving"]}
        self.assertEqual(moving, {"step_top_foot", "escalator_rail"})
        hand = next(c for c in report["contacts"] if c["contact"] == "escalator_rail")
        path = hand["anchor_path"]
        self.assertGreater(path[-1][0] - path[0][0], 0.1)
        motion = Motion(document)
        for frame_index in (3, 4):
            position = pose_world_positions(target, motion.frames[frame_index])["l_hand"]
            expected = path[frame_index - 2]
            error = sum((a - b) ** 2 for a, b in zip(position, expected)) ** 0.5
            self.assertLess(error, 0.005)

    def test_moving_platform_carries_root_and_feet(self):
        _, target, _, _, document = build("simple_target", "moving_platform_source", "simple_source_to_simple_target")
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        self.assertEqual(report["unmet"], [])
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])
        first_root = document["frames"][0]["root_translation"]
        last_root = document["frames"][-1]["root_translation"]
        self.assertGreater(last_root[0] - first_root[0], 0.1)
        contact = next(c for c in report["contacts"] if c["contact"] == "platform_left")
        self.assertGreater(contact["max_root_adjustment"], 0.1)

    def test_tall_target_keeps_pose_and_satisfies_contacts(self):
        _, target, _, _, document = build_railing("simple_tall_target", "simple_source_to_simple_tall_target")
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        self.assertEqual(report["unmet"], [])
        self.assertEqual(check_contact_sliding(target, Motion(document), tolerance=0.005), [])
        motion = Motion(document)
        for frame in motion.frames:
            positions = pose_world_positions(target, frame)
            self.assertGreater(positions["hips"][1], positions["l_foot"][1])
            self.assertGreater(positions["hips"][1], 0.5 * document["root_scale"])


class PriorityAndBlendTests(unittest.TestCase):
    def test_unsatisfiable_contact_yields_with_error_not_success(self):
        _, target, _, _, document = build_railing()
        for contact in document["contacts"]:
            if contact.get("name") == "escalator_rail":
                contact["ground_y"] = 0.2  # rail at ankle height: unreachable
                contact["velocity"] = [0.0, 0.0, 0.0]
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        by_name = {contact["contact"]: contact for contact in report["contacts"]}
        self.assertTrue(by_name["step_edge_foot"]["met"])
        self.assertTrue(by_name["step_top_foot"]["met"])
        self.assertFalse(by_name["escalator_rail"]["met"])
        self.assertGreater(by_name["escalator_rail"]["max_error"], 0.1)
        hand_unmet = [item for item in report["unmet"] if item["joint"] == "l_hand"]
        self.assertTrue(hand_unmet)
        self.assertTrue(all(item["yielded"] for item in hand_unmet))
        self.assertTrue(all(set(item["yielded_to"]) for item in hand_unmet))
        self.assertTrue(all("error" in item and "residual" in item for item in hand_unmet))

    def test_priority_order_decides_which_contact_yields(self):
        _, target, _, _, document = build(
            "simple_target", "walk_ground_source", "simple_source_to_simple_target"
        )
        # Both feet planted for the whole clip, but the left foot anchor is
        # dragged 1.5 m aside: only the higher-priority foot can be satisfied.
        scale = document["root_scale"]
        contacts = []
        for contact in document["contacts"]:
            contact["start"] = 0
            contact["end"] = 3
            contact["name"] = f"{contact['joint']}_hold"
            contacts.append(contact)
        left = next(c for c in contacts if c["joint"] == "l_foot")
        left_fk = pose_world_positions(target, document["frames"][0])["l_foot"]
        left["target_path"] = [
            [left_fk[0] + 1.5 * scale, 0.1119 * scale, left_fk[2]] for _ in range(4)
        ]
        document["contacts"] = contacts
        comparison = compare_schemes(
            target,
            document,
            [
                {"name": "right_first", "priorities": {"r_foot": 0, "l_foot": 1}},
                {"name": "left_first", "priorities": {"r_foot": 1, "l_foot": 0}},
            ],
            ground_scale=document["root_scale"],
        )
        schemes = {scheme["name"]: scheme for scheme in comparison["schemes"]}
        right_first = {c["joint"]: c for c in schemes["right_first"]["per_contact"]}
        left_first = {c["joint"]: c for c in schemes["left_first"]["per_contact"]}
        self.assertTrue(right_first["r_foot"]["met"])
        self.assertFalse(right_first["l_foot"]["met"])
        self.assertTrue(left_first["l_foot"]["met"])
        self.assertFalse(left_first["r_foot"]["met"])
        self.assertIn("l_foot", schemes["right_first"]["yielded_joints"])
        self.assertIn("r_foot", schemes["left_first"]["yielded_joints"])

    def test_blend_ramps_hand_motion_and_free_frames_stay_untouched(self):
        _, target, _, _, document = build_railing()
        original = json.loads(json.dumps(document))
        report = apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        hand = next(c for c in report["contacts"] if c["contact"] == "escalator_rail")
        anchor_in = hand["anchor_path"][0]
        original_hand = pose_world_positions(target, original["frames"][2])["l_hand"]
        blended_hand = pose_world_positions(target, document["frames"][2])["l_hand"]
        # blend_in weight at the first frame is 0.5: halfway from free pose to anchor
        for component in range(3):
            midpoint = (original_hand[component] + anchor_in[component]) / 2
            self.assertAlmostEqual(blended_hand[component], midpoint, places=3)

        _, target2, _, _, walk = build("simple_target", "walk_ground_source", "simple_source_to_simple_target")
        walk["contacts"] = [
            contact for contact in walk["contacts"]
            if contact["joint"] == "r_foot"
        ]
        untouched = json.loads(json.dumps(walk))
        apply_foot_contacts(target2, walk, ground_scale=walk["root_scale"])
        for frame_index in (2, 3):
            self.assertEqual(walk["frames"][frame_index], untouched["frames"][frame_index])

    def test_invalid_moving_contact_specs_are_rejected(self):
        joints = {"l_foot"}
        with self.assertRaises(ValueError):
            parse_contacts(
                [{"joint": "l_foot", "start": 0, "end": 2, "target_path": [[0, 0, 0]]}], 4, joints
            )
        with self.assertRaises(ValueError):
            parse_contacts(
                [{
                    "joint": "l_foot",
                    "start": 0,
                    "end": 1,
                    "target_path": [[0, 0, 0], [0, 0, 0]],
                    "velocity": [1, 0, 0],
                }],
                4,
                joints,
            )


class ManualPinHandoffTests(unittest.TestCase):
    def test_manual_pin_freezes_prefix_and_reports_moving_support_conflict(self):
        _, target, _, _, document = build_railing()
        apply_foot_contacts(target, document, ground_scale=document["root_scale"])
        confirmed = json.loads(json.dumps(document))
        pinned_position = pose_world_positions(target, document["frames"][3])["l_foot"]
        document["contacts"][1]["velocity"] = [2.5, 0.0, 0.0]
        document["contacts"][1].pop("anchor_path", None)

        report = apply_foot_contacts(
            target,
            document,
            ground_scale=document["root_scale"],
            manual_pins=[{
                "pin_id": "reviewer_keeps_left_foot",
                "joint": "l_foot",
                "start": 3,
                "end": 5,
                "position": list(pinned_position),
                "basis": "reviewer manually retained stair contact before platform transfer",
            }],
            freeze_before=3,
        )

        self.assertEqual(document["frames"][0], confirmed["frames"][0])
        self.assertEqual(document["frames"][1], confirmed["frames"][1])
        self.assertEqual(document["frames"][2], confirmed["frames"][2])
        self.assertNotEqual(document["frames"][3], confirmed["frames"][3])
        conflicts = [item for item in report["unmet"] if item["type"] == "manual_pin_conflict"]
        self.assertTrue(conflicts)
        self.assertEqual({item["frame"] for item in conflicts}, {3, 4, 5})
        conflict = next(item for item in conflicts if item["frame"] == 5)
        self.assertEqual(conflict["joint"], "l_foot")
        self.assertEqual(conflict["contact"], "step_top_foot")
        self.assertTrue(conflict["yielded"])
        self.assertGreater(conflict["error"], 0.05)

        audit = build_handoff_audit(target, document, report, solve_start=3, manual_pins=report["manual_pins"])
        frame_2 = next(item for item in audit["timeline"] if item["frame"] == 2)
        frame_4 = next(item for item in audit["timeline"] if item["frame"] == 4)
        self.assertEqual({item["joint"] for item in frame_2["supports"]}, {"r_foot", "l_foot", "l_hand"})
        self.assertTrue(frame_4["yielded"])
        self.assertIn("step_top_foot", audit["contact_histories"])
        self.assertEqual(
            audit["contact_histories"]["step_top_foot"]["current"]["yield_ranges"],
            [{"start": 3, "end": 5}],
        )

    def test_regeneration_preserves_manual_pin_basis_and_contact_history(self):
        _, target, _, _, first_document = build_railing()
        apply_foot_contacts(target, first_document, ground_scale=first_document["root_scale"])
        pinned_position = pose_world_positions(target, first_document["frames"][3])["l_foot"]
        first_document["manual_pins"] = [{
            "pin_id": "reviewer_keeps_left_foot",
            "joint": "l_foot",
            "start": 3,
            "end": 5,
            "position": list(pinned_position),
            "basis": "reviewer selected this exact stair contact",
            "freeze_before": 3,
        }]
        first_document["cyclic"] = True
        first_audit = build_handoff_audit(
            target,
            first_document,
            {"unmet": [], "contacts": []},
            solve_start=3,
            manual_pins=first_document["manual_pins"],
        )
        first_document["contact_handoffs"] = first_audit
        regenerated = json.loads(json.dumps(first_document))
        report = apply_foot_contacts(
            target,
            regenerated,
            ground_scale=regenerated["root_scale"],
            manual_pins=regenerated["manual_pins"],
            freeze_before=3,
        )
        audit = build_handoff_audit(
            target,
            regenerated,
            report,
            solve_start=3,
            manual_pins=report["manual_pins"],
        )
        self.assertEqual(audit["manual_pins"][0]["basis"], "reviewer selected this exact stair contact")
        self.assertEqual(audit["skeleton_scale"], regenerated["root_scale"])
        self.assertIn("step_edge_foot", audit["contact_histories"])
        self.assertIn("step_top_foot", audit["contact_histories"])
        self.assertIsNotNone(audit["cyclic_closure"])


class SchemeComparisonBatchTests(unittest.TestCase):
    def test_compare_batch_saves_reports_and_recomputes_identically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            batch_path = ROOT / "data" / "batches" / "constraint_schemes_batch.json"
            first = run_batch(RESOURCES, batch_path, output)
            self.assertEqual(first["failed"], 0, first)
            reports = {}
            for name in ("turn_scheme_check", "stairs_scheme_check", "railing_scheme_check", "platform_scheme_check"):
                reports[name] = (output / "comparisons" / f"{name}.json").read_text(encoding="utf-8")
            second = run_batch(RESOURCES, batch_path, output, overwrite=True)
            self.assertEqual(second["failed"], 0, second)
            for name, content in reports.items():
                recomputed = (output / "comparisons" / f"{name}.json").read_text(encoding="utf-8")
                self.assertEqual(json.loads(recomputed), json.loads(content), name)

            railing = json.loads(reports["railing_scheme_check"])
            scheme_names = {scheme["name"] for scheme in railing["schemes"]}
            self.assertEqual(scheme_names, {"feet_first", "hand_first", "no_rail"})
            for scheme in railing["schemes"]:
                self.assertIn("per_contact", scheme)
                self.assertIn("yielded_joints", scheme)
            self.assertEqual(railing["recommended"], "feet_first")

            platform = json.loads(reports["platform_scheme_check"])
            ride = next(s for s in platform["schemes"] if s["name"] == "ride_platform")
            self.assertTrue(all(c["moving"] for c in ride["per_contact"]))
            self.assertTrue(ride["passed"])
            platform_audit = platform["handoff_audits"]["ride_platform"]
            self.assertTrue(all(item["direction"] == "positive_x" for item in platform_audit["moving_support_checks"]))
            self.assertIn("platform_left", platform_audit["contact_histories"])

            motion = load_motion(output / "motions" / "railing_simple_target.json")
            target = load_skeleton(RESOURCES / "skeletons" / "simple_target.json")
            self.assertEqual(check_contact_sliding(target, motion, tolerance=0.005), [])
            platform_motion = load_motion(output / "motions" / "moving_platform_target.json")
            self.assertEqual(check_contact_sliding(target, platform_motion, tolerance=0.005), [])
            railing_document = json.loads((output / "motions" / "railing_simple_target.json").read_text(encoding="utf-8"))
            self.assertIn("step_edge_foot", railing_document["contact_handoffs"]["contact_histories"])

    def test_selected_scheme_motion_records_grounding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            summary = run_batch(RESOURCES, ROOT / "data" / "batches" / "constraint_schemes_batch.json", output)
            job = next(item for item in summary["jobs"] if item["job"] == "railing_scheme_check")
            self.assertEqual(job["status"], "success")
            grounding = job["validation"]["grounding"]
            names = {contact["contact"] for contact in grounding["contacts"]}
            self.assertEqual(names, {"step_edge_foot", "step_top_foot", "escalator_rail"})
            self.assertTrue(all(contact["met"] for contact in grounding["contacts"]))


if __name__ == "__main__":
    unittest.main()
