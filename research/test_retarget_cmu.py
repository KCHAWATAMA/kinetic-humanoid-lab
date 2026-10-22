import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import yaml

from cmu import CACHE, forward, load
from fetch_assets import MODEL_DIR
from retarget_cmu import PROTOCOL, contacts, filtered, generate, lock_feet, quaternion, run, slerp, spans
from rollout import load_model, verify_assets


class RetargetGeometryTests(unittest.TestCase):
    def test_contact_gap_closing_and_short_stance_rejection(self):
        raw = np.array([False, True, True, False, True, True, False, False, True, False])
        cleaned = contacts(raw, fps=10, maximum_gap_s=.1, minimum_stance_s=.2)
        self.assertEqual(spans(cleaned), [(1, 6)])
        self.assertFalse(cleaned[0])
        self.assertFalse(cleaned[-1])

    def test_filter_preserves_constants_and_interior_linear_path(self):
        np.testing.assert_allclose(filtered(np.ones((15, 3))), 1)
        path = np.arange(15.)[:, None]
        np.testing.assert_allclose(filtered(path)[3:-3], path[3:-3])

    def test_stance_lock_is_stationary_and_swing_endpoints_are_continuous(self):
        raw = np.zeros((21, 3))
        raw[:, 0] = np.linspace(0, .2, 21)
        raw[:, 2] = .05 + .08 * np.sin(np.linspace(0, np.pi, 21))
        stance = np.zeros(21, dtype=bool)
        stance[:5] = stance[-5:] = True
        feet, yaw, correction = lock_feet(raw, np.linspace(-.1, .1, 21), stance, .035)
        np.testing.assert_allclose(np.diff(feet[:5], axis=0), 0)
        np.testing.assert_allclose(np.diff(feet[-5:], axis=0), 0)
        np.testing.assert_allclose(feet[stance, 2], .035)
        self.assertGreater(feet[10, 2], .06)
        self.assertLess(np.max(np.linalg.norm(np.diff(feet, axis=0), axis=1)), .025)
        self.assertLess(correction, .15)
        self.assertEqual(yaw[0], yaw[4])
        with self.assertRaisesRegex(ValueError, "No source stance"):
            lock_feet(raw, yaw, np.zeros(21, dtype=bool), .035)

    def test_quaternion_interpolation_uses_short_arc_and_unit_length(self):
        a = quaternion(np.eye(3))
        b = quaternion(np.diag([-1., -1., 1.]))
        midpoint = slerp(a, b, .5)
        np.testing.assert_allclose(np.abs(midpoint), [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-12)
        np.testing.assert_allclose(slerp(a, -a, .5), a, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(midpoint), 1)

    def test_existing_evidence_is_never_overwritten_and_errors_are_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evidence.json"
            output.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Path(folder) / "missing-assets", CACHE, output)
            self.assertEqual(output.read_text(), "original")
            failed = Path(folder) / "failed.json"
            with self.assertRaises(FileNotFoundError):
                run(Path(folder) / "missing-assets", CACHE, failed)
            record = json.loads(failed.read_text())
            self.assertEqual(record["status"], "error")
            self.assertFalse(record["physicsExecuted"])
            self.assertIn("FileNotFoundError", record["error"])


@unittest.skipUnless((CACHE / "provenance.json").exists() and Path(".research-cache/protomotions/provenance.json").exists(),
                     "Fetch pinned model and selected CMU source for retarget integration checks.")
class RetargetReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = Path(".research-cache/protomotions")
        verify_assets(assets)
        directory = assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        cls.model = load_model(assets, cls.meta, config)
        cls.protocol = json.loads(PROTOCOL.read_text())
        cls.frames, cls.diagnostics, _ = generate(cls.model, cls.meta, np.asarray(config["robot"]["default_dof_pos"]), cls.protocol)

    def test_reproduce_saved_failure_without_inference_or_gate_relaxation(self):
        saved = json.loads(Path("public/mocap/g1-cmu-07-04-v1.json").read_text())
        self.assertEqual(saved["status"], "reference_failed")
        self.assertFalse(saved["physicsExecuted"])
        self.assertEqual(saved["protocol"], self.protocol)
        self.assertEqual(saved["reference"]["gates"], self.diagnostics["gates"])
        self.assertFalse(self.diagnostics["passed"])
        for name, value in saved["reference"]["metrics"].items():
            np.testing.assert_allclose(self.diagnostics["metrics"][name], value, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(saved["referenceEvidence"]["jointPositions"], [f["q"] for f in self.frames], atol=1e-8)
        json.dumps(self.diagnostics, allow_nan=False)

    def test_actual_fk_limits_tendons_and_lookahead_hold(self):
        data = mujoco.MjData(self.model)
        violations = 0
        for frame in self.frames:
            data.qpos[:] = frame["qpos"]
            mujoco.mj_forward(self.model, data)
            np.testing.assert_allclose(data.xpos[1:], frame["body"], atol=1e-12)
            self.assertTrue(np.all(data.qpos[7:] >= self.model.jnt_range[1:, 0]))
            self.assertTrue(np.all(data.qpos[7:] <= self.model.jnt_range[1:, 1]))
            violations += bool(np.any(data.ten_length < self.model.tendon_range[:, 0] - 1e-6)
                               or np.any(data.ten_length > self.model.tendon_range[:, 1] + 1e-6))
        self.assertEqual(violations, self.diagnostics["metrics"]["tendon_violation_frames"])
        self.assertGreater(violations, 0)
        end = round(self.protocol["seconds"] / self.protocol["control_dt"])
        self.assertEqual(len(self.frames), end + 9)
        for frame in self.frames[end + 1:]:
            np.testing.assert_array_equal(frame["qpos"], self.frames[end]["qpos"])
            np.testing.assert_array_equal(frame["v"], np.zeros(29))

    def test_joint_velocities_and_reference_slip_are_measured_not_requested(self):
        poses = np.asarray([frame["qpos"] for frame in self.frames])
        expected_velocity = np.gradient(poses[:, 7:], .02, axis=0)
        np.testing.assert_allclose([frame["v"] for frame in self.frames], expected_velocity, atol=3e-6)
        feet = np.asarray([frame["feet"] for frame in self.frames])
        targets = np.asarray([frame["requestedFeet"] for frame in self.frames])
        error = np.linalg.norm(feet - targets, axis=2).max()
        self.assertAlmostEqual(error, self.diagnostics["metrics"]["max_foot_position_error_m"], places=10)
        stance = np.asarray([frame["stance"] for frame in self.frames])
        slips = np.linalg.norm(np.diff(feet[:, :, :2], axis=0) / .02, axis=2)[stance[1:] & stance[:-1]]
        self.assertAlmostEqual(float(slips.max()), self.diagnostics["metrics"]["maximum_stance_slip_m_s"], places=10)

    def test_contact_height_rejects_slow_late_right_foot_before_robot_ik(self):
        skeleton, frames, _ = load()
        positions = [forward(skeleton, frame)[0] for frame in frames]
        sole = filtered(np.asarray([(pose["rfoot"] + pose["rtoes"]) / 2 for pose in positions]))
        threshold = np.percentile(sole[:, 1], 5) + .045
        speed = np.linalg.norm(np.gradient(sole[:, [0, 2]], 1 / 120, axis=0), axis=1)
        index = round(3.1 * 120)
        self.assertAlmostEqual(sole[index, 1], .070822872670444, places=10)
        self.assertAlmostEqual(threshold, .05516963880964267, places=10)
        self.assertGreater(sole[index, 1], threshold)
        self.assertLess(speed[index], .5)


if __name__ == "__main__":
    unittest.main()
