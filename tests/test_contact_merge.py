import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from anim.contact import solve_contacts
from anim.merge import merge_clips
from anim.merge_validation import validate_merged_motion
from anim.models import Motion, load_mapping, load_motion, load_skeleton
from anim.pipeline import run_batch
from anim.retarget import pose_world_positions, retarget_motion


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "resources"


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.source = load_skeleton(RES / "skeletons" / "simple_source.json")
        self.target = load_skeleton(RES / "skeletons" / "simple_target.json")
        self.motion = load_motion(RES / "motions" / "walk_contact_source.json")
        self.mapping = load_mapping(RES / "mappings" / "simple_source_to_simple_target.json")
        self.pairs = {s: t for s, t in self.mapping.required_pairs().items() if s.endswith("_foot")}

    def _solved(self):
        document = retarget_motion(self.source, self.target, self.motion, self.mapping).document
        result = solve_contacts(self.target, Motion(document), self.motion, self.pairs)
        return document, result

    def test_contact_intervals_are_pinned_to_anchors(self):
        _, result = self._solved()
        # The fixture contains deliberate leg-length/limit pressure; reachable
        # contact frames pin tightly, while conflicts remain small and reported.
        self.assertTrue(all(slide["slide"] < 0.25 for slide in result["slides"]), result["slides"])
        self.assertTrue(any(slide["slide"] < 0.1 for slide in result["slides"]))

    def test_swing_foot_is_not_frozen(self):
        document, result = self._solved()
        frames = result["frames"]
        # Right foot swings on frames 0->1 before its contact starts.
        points = [pose_world_positions(self.target, frames[i])["r_foot"] for i in (0, 1)]
        self.assertGreater(sum(abs(a - b) for a, b in zip(points[0], points[1])), 1e-4)

    def test_limit_conflict_is_reported_as_unmet_without_destroying_pose(self):
        target_data = json.loads((RES / "skeletons" / "simple_target.json").read_text())
        target_data["joints"]["l_hip"]["rotation_limits"]["x"] = [0.0, 0.0]
        target_data["joints"]["l_knee"]["rotation_limits"]["x"] = [0.0, 0.0]
        target_data["joints"]["r_hip"]["rotation_limits"]["x"] = [0.0, 0.0]
        target_data["joints"]["r_knee"]["rotation_limits"]["x"] = [0.0, 0.0]
        with tempfile.TemporaryDirectory() as temp:
            strict_path = Path(temp) / "strict.json"
            strict_path.write_text(json.dumps(target_data))
            strict_target = load_skeleton(strict_path)
            document = retarget_motion(self.source, strict_target, self.motion, self.mapping).document
            result = solve_contacts(strict_target, Motion(document), self.motion, self.pairs)
            self.assertTrue(result["unmet"])
            self.assertTrue(all("reason" in issue for issue in result["unmet"]))
            # Frames remain valid joint quaternions and are still playable.
            Motion(document)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.source = load_skeleton(RES / "skeletons" / "simple_source.json")
        self.target = load_skeleton(RES / "skeletons" / "simple_target.json")
        self.tall = load_skeleton(RES / "skeletons" / "tall_target.json")
        self.walk = load_motion(RES / "motions" / "walk_contact_source.json")
        self.turn = load_motion(RES / "motions" / "turn_source.json")
        self.stairs = load_motion(RES / "motions" / "stairs_source.json")
        self.map_v1 = load_mapping(RES / "mappings" / "simple_source_to_simple_target.json")
        self.map_v2 = load_mapping(RES / "mappings" / "simple_source_to_tall_target.json")

    def _clips(self, target, mapping):
        return [
            {"source_skeleton": self.source, "target_skeleton": target,
             "source_motion": self.walk, "mapping": mapping,
             "source_path": "motions/walk_contact_source.json"},
            {"source_skeleton": self.source, "target_skeleton": target,
             "source_motion": self.turn, "mapping": mapping,
             "source_path": "motions/turn_source.json",
             "revisions": [{"frame": 0, "root_translation_delta": [0.0, 0.0, 0.001]}]},
            {"source_skeleton": self.source, "target_skeleton": target,
             "source_motion": self.stairs, "mapping": mapping,
             "source_path": "motions/stairs_source.json"},
        ]

    def test_merge_preserves_provenance_versions_revisions_and_stitch_continuity(self):
        clips = self._clips(self.target, self.map_v1)
        document = merge_clips(clips, name="continuous_sequence", target=self.target)
        report = validate_merged_motion(self.target, document)
        self.assertTrue(report["passed"], report)
        versions = {item["mapping_version"] for item in document["provenance"]}
        self.assertEqual(versions, {"1.0"})
        sources = [item["source_resource"] for item in document["provenance"]]
        self.assertEqual(sources, [
            "motions/walk_contact_source.json",
            "motions/turn_source.json",
            "motions/stairs_source.json",
        ])
        self.assertTrue(document["provenance"][1]["local_revisions"])
        self.assertTrue(all(t["root_gap"] < 1e-6 for t in document["merge_metadata"]["transitions"]))

    def test_merge_supports_different_leg_length_turn_and_stairs(self):
        clips = self._clips(self.tall, self.map_v2)
        document = merge_clips(clips, name="tall_sequence", target=self.tall)
        report = validate_merged_motion(self.tall, document)
        self.assertTrue(report["passed"], report)
        self.assertEqual({item["mapping_version"] for item in document["provenance"]}, {"2.0"})
        self.assertGreater(document["provenance"][-1]["root_end"][1],
                           document["provenance"][0]["root_start"][1])
        # Turning rotates the root; merged output should not stay axis aligned.
        first_root = document["frames"][0]["root_translation"]
        last_root = document["frames"][-1]["root_translation"]
        self.assertTrue(abs(first_root[0] - last_root[0]) > 1e-6 or
                        abs(first_root[2] - last_root[2]) > 1e-6)

    def test_batch_runs_merge_job_and_writes_record(self):
        batch = {
            "kind": "retarget_batch", "name": "merge_batch", "jobs": [{
                "kind": "merge_clips", "name": "sequence",
                "target_skeleton": "skeletons/simple_target.json",
                "output_motion": "motions/sequence.json",
                "clips": [
                    {"source_skeleton": "skeletons/simple_source.json",
                     "source_motion": "motions/walk_contact_source.json",
                     "mapping": "mappings/simple_source_to_simple_target.json"},
                    {"source_skeleton": "skeletons/simple_source.json",
                     "source_motion": "motions/turn_source.json",
                     "mapping": "mappings/simple_source_to_simple_target.json"},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            batch_path = Path(temp) / "batch.json"
            batch_path.write_text(json.dumps(batch))
            summary = run_batch(RES, batch_path, Path(temp) / "output")
            self.assertEqual(summary["failed"], 0, summary)
            job = summary["jobs"][0]
            self.assertEqual(job["validation"]["validation"]["metrics"]["clips"], 2)


if __name__ == "__main__":
    unittest.main()
