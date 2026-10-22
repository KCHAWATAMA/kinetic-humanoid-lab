"""Reference coherence and physical locomotion metrics, not assumed walking success."""

import json
import hashlib
from pathlib import Path
import unittest

import mujoco
import numpy as np
import yaml

from fetch_assets import MODEL_DIR
from locomotion import FEET, PROTOCOL_PATH, FootStepCounter, LocomotionObserver, contact_telemetry, generate_reference, run_attempt
from rollout import grounded_root_height, load_model, make_inputs, run_case, verify_assets


class LocomotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = Path(".research-cache/protomotions")
        verify_assets(assets)
        directory = assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text(encoding="utf-8"))
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text(encoding="utf-8"))
        cls.default = np.asarray(config["robot"]["default_dof_pos"])
        cls.model = load_model(assets, cls.meta, config)
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.frames, cls.diagnostics = generate_reference(cls.model, cls.meta, cls.default, cls.protocol)

    def test_reference_is_joint_limited_ground_consistent_and_continuous(self):
        self.assertEqual(len(self.frames), 409)
        np.testing.assert_allclose(self.frames[0]["q"], self.default, atol=1e-7)
        self.assertAlmostEqual(self.diagnostics["target_forward_progress_m"], 0.35)
        self.assertLess(self.diagnostics["max_foot_position_error_m"], 0.0002)
        self.assertLess(self.diagnostics["max_foot_orientation_error_rad"], 0.001)
        self.assertLess(self.diagnostics["max_joint_velocity_rad_s"], 4)
        joints = np.stack([frame["q"] for frame in self.frames])
        self.assertTrue(np.all(joints >= self.model.jnt_range[1:, 0]))
        self.assertTrue(np.all(joints <= self.model.jnt_range[1:, 1]))
        bodies = [self.model.body(name).id - 1 for name in FEET]
        for previous, frame in zip(self.frames, self.frames[1:]):
            for side, body in enumerate(bodies):
                if previous["stance"][side] and frame["stance"][side]:
                    np.testing.assert_allclose(previous["feet"][side], frame["feet"][side], atol=1e-10)
                    np.testing.assert_allclose(previous["body"][body], frame["body"][body], atol=2e-6)
        expected_velocity = np.gradient(joints.astype(float), 0.02, axis=0, edge_order=2)
        np.testing.assert_allclose([frame["v"] for frame in self.frames], expected_velocity, atol=1e-5)

    def test_world_reference_is_not_smuggled_into_the_policy(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        inputs = make_inputs(data, self.meta, [self.frames[index] for index in (1, 2, 4, 8)], np.zeros(29))
        self.assertEqual(len(inputs), 8)
        self.assertNotIn("current_anchor_pos", inputs)
        self.assertNotIn("mimic_future_anchor_pos", inputs)
        self.assertNotIn("current_root_linear_vel", inputs)
        changed = [{**frame, "root": frame["root"] + 100, "body": frame["body"] + 100} for frame in self.frames[:9]]
        inputs_changed = make_inputs(data, self.meta, [changed[index] for index in (1, 2, 4, 8)], np.zeros(29))
        for key in inputs:
            np.testing.assert_array_equal(inputs[key], inputs_changed[key])

    def test_final_variant_preserves_gates_and_satisfies_reference_contract(self):
        variant = json.loads(PROTOCOL_PATH.with_name("locomotion-protocol-02.json").read_text(encoding="utf-8"))
        self.assertEqual(variant["criteria"], self.protocol["criteria"])
        self.assertEqual(variant["step_detection"], self.protocol["step_detection"])
        self.assertEqual(variant["reference_gates"], self.protocol["reference_gates"])
        frames, diagnostics = generate_reference(self.model, self.meta, self.default, variant)
        self.assertEqual(len(frames), 409)
        self.assertAlmostEqual(diagnostics["target_forward_progress_m"], 0.35)
        self.assertLess(diagnostics["max_foot_position_error_m"], 0.0002)
        self.assertLess(diagnostics["max_joint_velocity_rad_s"], 4)

    def test_contact_slip_is_world_contact_point_velocity(self):
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [0, 0, grounded_root_height(self.model, self.default) - 0.006]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = self.default
        data.qvel[0] = 0.2
        mujoco.mj_forward(self.model, data)
        force, speeds = contact_telemetry(self.model, data)
        self.assertTrue(np.all(force > 5))
        np.testing.assert_allclose(speeds, [0.2, 0.2], atol=1e-10)
        data.qpos[2] += 0.5
        mujoco.mj_forward(self.model, data)
        force, _ = contact_telemetry(self.model, data)
        np.testing.assert_array_equal(force, [0, 0])

    def test_step_counter_rejects_airborne_and_sliding_false_positives(self):
        counter = FootStepCounter(self.protocol["step_detection"])
        for t in np.arange(0, 1, 0.02):
            counter.sample(float(t), np.array([t, 0, 0.1]), False)
        self.assertEqual(counter.steps, 0)
        for t in np.arange(1, 2, 0.02):
            counter.sample(float(t), np.array([t, 0, 0.035]), True)
        self.assertEqual(counter.steps, 0)
        counter.sample(2.0, np.array([2.0, 0, 0.035]), True)
        counter.sample(2.02, np.array([2.02, 0, 0.07]), False)
        counter.sample(2.12, np.array([2.10, 0, 0.08]), False)
        counter.sample(2.22, np.array([2.15, 0, 0.035]), True)
        self.assertEqual(counter.steps, 1)

    def test_no_contacts_and_no_progress_cannot_pass(self):
        observer = LocomotionObserver(self.protocol, 0.02)
        observer.samples = [{
            "time": t, "root": [0, 0, 0.75], "referenceRoot": [0.35 * t / 8, 0, 0.75],
            "loaded": [False, False], "contactSlipMS": [0, 0],
        } for t in (0, 8)]
        result = observer.summarize({"completed": True, "simulated_seconds": 8, "fell": False,
                                     "numerical_warnings": 0, "joint_rmse_rad": 0, "mean_body_position_error_m": 0})
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["forward_progress"])
        self.assertFalse(result["gates"]["verified_steps"])
        self.assertFalse(result["gates"]["supported_motion"])
        self.assertFalse(result["gates"]["mean_contact_slip"])
        self.assertEqual(result["metrics"]["mean_contact_slip_m_s"], [None, None])

    def test_custom_reference_cannot_change_the_physical_initial_root(self):
        frames = [{**frame, "body": frame["body"].copy()} for frame in self.frames]
        frames[0]["body"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "initial state"):
            run_case(self.model, None, self.meta, self.default, "slow_walk", 8, controller="direct_pd", reference_frames=frames)
        with self.assertRaisesRegex(ValueError, "lookahead"):
            run_case(self.model, None, self.meta, self.default, "slow_walk", 8, controller="direct_pd", reference_frames=frames[:10])

    def test_saved_attempts_recompute_metrics_without_cherry_picking(self):
        directory = Path("public/locomotion")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual([item["attempt"] for item in summary["attempts"]], [1, 2])
        self.assertEqual(summary["walkingAchieved"], any(item["passed"] for item in summary["attempts"]))
        for item in summary["attempts"]:
            payload = (directory / item["artifact"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["artifactSha256"])
            attempt = json.loads(payload)
            self.assertEqual(attempt["status"], "complete")
            physics = attempt["physics"]["metrics"]
            self.assertEqual((physics["controller_calls"], physics["physics_steps"], physics["simulated_seconds"]), (400, 8000, 8))
            self.assertEqual(len(attempt["physics"]["frames"]), 201)
            observer = LocomotionObserver(attempt["protocol"], 0.02)
            observer.samples = attempt["locomotion"]["samples"]
            self.assertEqual(len(observer.samples), 401)
            for sample in observer.samples:
                self.assertEqual(sample["loaded"], [force >= 5 for force in sample["normalForceN"]])
                for side in range(2):
                    observer.counters[side].sample(sample["time"], np.asarray(sample["feet"][side]), sample["loaded"][side])
            recomputed = observer.summarize(physics)
            self.assertEqual(recomputed, attempt["locomotion"])
            self.assertEqual(item["metrics"], recomputed["metrics"])
            self.assertEqual(item["gates"], recomputed["gates"])
            self.assertEqual(item["passed"], all(item["gates"].values()))

    def test_attempts_cannot_be_overwritten_or_extended_beyond_bound(self):
        with self.assertRaises(FileExistsError):
            run_attempt(Path(".research-cache/protomotions"), Path("public/locomotion"), 1)
        with self.assertRaisesRegex(ValueError, "only attempts"):
            run_attempt(Path(".research-cache/protomotions"), Path("public/locomotion"), 3)


if __name__ == "__main__":
    unittest.main()
