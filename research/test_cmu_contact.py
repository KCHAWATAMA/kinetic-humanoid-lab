import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from cmu_contact import PROTOCOL, fit_valid, generate, grounded_rocking_feet, infer_contacts, source_targets
from export_mocap import digest
from fetch_assets import MODEL_DIR
from locomotion import LocomotionObserver
from rollout import load_model, run_case, xyzw


class ContactGeometryTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL.read_text())

    def test_marker_height_is_not_mistaken_for_sole_height(self):
        sole = np.tile([0., 0., .071], (30, 1))
        self.assertTrue(infer_contacts(sole, 120, self.protocol["retarget"]).all())
        sole[:, 0] = np.arange(30) / 120
        self.assertFalse(infer_contacts(sole, 120, self.protocol["retarget"]).any())
        sole[:, 0] = 0
        sole[:, 2] = np.arange(30) / 120
        self.assertFalse(infer_contacts(sole, 120, self.protocol["retarget"]).any())

    def test_rocking_capsules_are_grounded_without_heel_to_toe_jump(self):
        angles = np.linspace(-.4, .4, 101)
        rotations = Rotation.from_euler("y", angles).as_matrix()
        points = np.array([[-.05, 0, -.025], [.13, 0, -.025]])
        radii = np.array([.01, .01])
        feet = grounded_rocking_feet(np.tile([0., 0., .05], (101, 1)), rotations,
                                    np.ones(101, dtype=bool), points, radii, .002)
        bottoms = [min((points @ r.T + p)[:, 2] - radii) for r, p in zip(rotations, feet)]
        np.testing.assert_allclose(bottoms, .002, atol=1e-12)
        self.assertLess(np.linalg.norm(np.diff(feet, axis=0), axis=1).max(), .002)
        self.assertGreater(np.ptp(feet[:, 2]), .02)

    def test_physical_criteria_unchanged_and_bad_warm_start_not_propagated(self):
        previous = json.loads(Path("research/cmu-protocol.json").read_text())
        for key in ("criteria", "step_detection", "progress_gate_ratio", "seconds", "control_dt"):
            self.assertEqual(previous[key], self.protocol[key])
        invalid = {"optimizer_success": False}
        self.assertFalse(fit_valid(invalid, self.protocol["reference_gates"]))
        with patch("cmu_contact.source_targets", return_value=([None, None], None, {}, {})), patch("cmu_contact.ConstrainedIK") as factory:
            factory.return_value.solve.return_value = (None, None, invalid)
            frames, report, _ = generate(None, {}, np.zeros(29), self.protocol, lambda *_: None)
            self.assertEqual(factory.return_value.solve.call_count, 1)
            self.assertEqual(frames, [])
            self.assertFalse(report["passed"])
            self.assertIn("not propagated", report["error"])


class ContactSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = Path(".research-cache/protomotions")
        directory = assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        cls.default = np.asarray(config["robot"]["default_dof_pos"])
        cls.model = load_model(assets, cls.meta, config)
        cls.protocol = json.loads(PROTOCOL.read_text())

    def test_same_source_timing_has_late_support_and_real_foot_rocking(self):
        targets, stance, summary, source = source_targets(self.model, self.default, self.protocol)
        self.assertEqual(len(targets), 187)
        self.assertEqual(source["subject"], "07")
        self.assertEqual(source["trial"], "04")
        self.assertTrue(stance[155, 1])
        self.assertGreater(summary["source_support_fraction"], .85)
        pitches = np.asarray([Rotation.from_matrix(t.foot_rotations[0]).as_euler("ZYX")[1] for t in targets])
        self.assertGreater(np.ptp(pitches), 1.)
        self.assertTrue(np.isfinite([target.feet for target in targets]).all())

    def test_explicit_initial_state_is_checked_and_not_reassigned_during_rollout(self):
        data = mujoco.MjData(self.model)
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = [0, 0, .9], [1, 0, 0, 0], self.default
        data.qvel[0] = .4
        mujoco.mj_forward(self.model, data)
        initial = data.qpos.copy(), data.qvel.copy()
        frame = {"q": self.default.astype(np.float32), "v": np.zeros(29, dtype=np.float32),
                 "anchor": xyzw(data.xquat[self.meta["robot"]["anchor_body_index"] + 1]),
                 "body": data.xpos[1:].copy()}
        references = [frame] * 59
        samples = []
        result = run_case(self.model, None, self.meta, self.default, "moving-initialization-test", 1,
                          controller="direct_pd", reference_frames=references, initial_state=initial,
                          observe=lambda _, d, i, r: samples.append((d.qpos.copy(), d.qvel.copy())))
        np.testing.assert_array_equal(samples[0][0], initial[0])
        np.testing.assert_array_equal(samples[0][1], initial[1])
        self.assertGreater(abs(samples[-1][0][0]), .02)
        self.assertTrue(result["metrics"]["completed"])
        wrong = initial[0].copy()
        wrong[0] += .1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            run_case(self.model, None, self.meta, self.default, "bad", 1, controller="direct_pd",
                     reference_frames=references, initial_state=(wrong, initial[1]))
        with self.assertRaisesRegex(ValueError, "requires matching"):
            run_case(self.model, None, self.meta, self.default, "stand", 1, controller="direct_pd", initial_state=initial)

    def test_saved_reference_satisfies_actual_geometry_and_temporal_constraints(self):
        result = json.loads(Path("public/mocap-v2/attempt-03.json").read_text())
        self.assertTrue(result["reference"]["passed"])
        frames = result["referenceEvidence"]
        poses, velocities = np.asarray(frames["qpos"]), np.asarray(frames["qvel"])
        self.assertEqual(poses.shape, (195, 36))
        self.assertLessEqual(np.abs(velocities[:, 6:]).max(), 8)
        np.testing.assert_allclose(velocities[:, 6:], np.gradient(poses[:, 7:], .02, axis=0), atol=1e-10)
        np.testing.assert_array_equal(poses[187:], np.repeat(poses[186:187], 8, axis=0))
        data = mujoco.MjData(self.model)
        for q, body in zip(poses, frames["body"]):
            data.qpos[:] = q
            mujoco.mj_forward(self.model, data)
            np.testing.assert_allclose(data.xpos[1:], body, atol=1e-10)
            self.assertTrue(np.all(q[7:] >= self.model.jnt_range[1:, 0]))
            self.assertTrue(np.all(q[7:] <= self.model.jnt_range[1:, 1]))
            self.assertTrue(np.all(data.ten_length >= self.model.tendon_range[:, 0] - 1e-8))
            self.assertTrue(np.all(data.ten_length <= self.model.tendon_range[:, 1] + 1e-8))
        for report in result["reference"]["solver"]:
            fit = report["starts"][-1]
            self.assertTrue(fit_valid(fit, self.protocol["reference_gates"]))
            self.assertLessEqual(max(abs(v) for v in fit["root_adjustment_m"]), .12)

    def test_saved_physical_steps_fail_the_unchanged_world_tracking_gates(self):
        result = json.loads(Path("public/mocap-v2/attempt-03.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["physicsExecuted"])
        physics = result["physics"]["metrics"]
        self.assertEqual((physics["controller_calls"], physics["physics_steps"], physics["error_samples"]), (186, 3720, 187))
        self.assertFalse(physics["fell"])
        self.assertEqual(physics["numerical_warnings"], 0)
        observer = LocomotionObserver(result["physicalProtocol"], .02)
        observer.samples = result["locomotion"]["samples"]
        for sample in observer.samples:
            np.testing.assert_array_equal(sample["loaded"], np.asarray(sample["normalForceN"]) >= 5)
            for side, counter in enumerate(observer.counters):
                counter.sample(sample["time"], np.asarray(sample["feet"][side]), sample["loaded"][side])
        computed = observer.summarize(physics)
        self.assertEqual(computed, result["locomotion"])
        self.assertEqual(computed["metrics"]["verified_steps_per_foot"], [2, 2])
        self.assertGreater(computed["metrics"]["forward_progress_m"], 2)
        self.assertFalse(computed["passed"])
        self.assertEqual([name for name, passed in computed["gates"].items() if not passed],
                         ["lateral_drift", "root_tracking", "body_tracking"])
        ledger = json.loads(Path("public/mocap-v2/execution-ledger.json").read_text())
        self.assertEqual(ledger["physicalTrials"], 1)
        for name, expected in ledger["historicalSha256"].items():
            self.assertEqual(digest(Path(name)), expected)
        for entry in ledger["executions"]:
            self.assertEqual(digest(Path("public/mocap-v2") / entry["artifact"]), entry["sha256"])
        replay = json.loads(Path("public/mocap-v2/physics-replay.json").read_text())
        self.assertEqual(replay["cases"][0], result["physics"])
        self.assertEqual(replay["sourceArtifactSha256"], digest(Path("public/mocap-v2/attempt-03.json")))


if __name__ == "__main__":
    unittest.main()
