"""Offline contract checks after fetching assets and installing requirements."""

from pathlib import Path
import json
import unittest

import mujoco
import numpy as np
import yaml

from fetch_assets import MODEL_DIR, MODEL_SHA256, REVISION
from rollout import grounded_root_height, load_model, make_inputs, make_session, reference_pose, verify_assets, xyzw


class DeploymentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(".research-cache/protomotions")
        verify_assets(cls.assets)
        directory = cls.assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text(encoding="utf-8"))
        cls.config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text(encoding="utf-8"))
        cls.default = np.asarray(cls.config["robot"]["default_dof_pos"])
        cls.model = load_model(cls.assets, cls.meta, cls.config)
        cls.session = make_session(directory / "compiled_models/unified_pipeline.onnx", cls.meta)

    def test_robot_contract_and_torque_limits(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (36, 35, 29))
        self.assertAlmostEqual(self.model.opt.timestep, 0.001)
        self.assertEqual(self.model.nmesh, 0)
        self.assertTrue(np.all(self.model.actuator_forcelimited))
        self.assertTrue(np.all(self.model.actuator_forcerange[:, 1] > 0))
        self.assertAlmostEqual(self.model.opt.gravity[2], -9.81)

    def test_generated_reference_is_stationary_at_start_and_smooth(self):
        names = self.meta["robot"]["joint_names"]
        q, v = reference_pose(self.default, names, 0, "arm_sweep")
        np.testing.assert_array_equal(q, self.default)
        np.testing.assert_array_equal(v, np.zeros(29))
        for t in (0.1, 1.0, 2.5):
            q1, _ = reference_pose(self.default, names, t + 1e-5, "arm_sweep")
            q0, _ = reference_pose(self.default, names, t - 1e-5, "arm_sweep")
            _, v = reference_pose(self.default, names, t, "arm_sweep")
            np.testing.assert_allclose((q1 - q0) / 2e-5, v, atol=1e-8)

    def test_observation_shapes_and_real_policy_output(self):
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [0, 0, grounded_root_height(self.model, self.default)]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = self.default
        mujoco.mj_forward(self.model, data)
        anchor = xyzw(data.xquat[self.meta["robot"]["anchor_body_index"] + 1])
        future = [{"q": self.default, "v": np.zeros(29), "anchor": anchor}] * 4
        previous = np.arange(29, dtype=np.float32) / 100
        inputs = make_inputs(data, self.meta, future, previous)
        for declared in self.meta["policy_inputs"]:
            self.assertEqual(list(inputs[declared["name"]].shape), declared["shape"])
            self.assertEqual(inputs[declared["name"]].dtype, np.float32)
        np.testing.assert_array_equal(inputs["historical_processed_actions"][0, 0], previous)
        outputs = self.session.run(None, inputs)
        self.assertTrue(all(np.isfinite(value).all() for value in outputs))
        self.assertEqual(outputs[1].shape, (1, 29))
        np.testing.assert_allclose(outputs[2].reshape(-1), self.meta["control"]["stiffness"], rtol=1e-5)

    def test_quaternion_boundary_conversion(self):
        np.testing.assert_array_equal(xyzw(np.array([1, 0, 0, 0])), [0, 0, 0, 1])
        np.testing.assert_array_equal(xyzw(np.array([[0, 1, 0, 0]])), [[1, 0, 0, 0]])

    def test_recorded_checkpoint_meets_smoke_thresholds(self):
        replay = json.loads(Path("public/physics-replay.json").read_text(encoding="utf-8"))
        self.assertEqual(replay["kind"], "recorded-pretrained-policy-physics")
        self.assertEqual(replay["provenance"]["revision"], REVISION)
        model_record = next(item for item in replay["provenance"]["files"] if item["path"].endswith(".onnx"))
        self.assertEqual(model_record["sha256"], MODEL_SHA256)
        self.assertEqual([case["name"] for case in replay["cases"]], ["stand", "arm_sweep"])
        for case in replay["cases"]:
            with self.subTest(case=case["name"]):
                metrics = case["metrics"]
                self.assertTrue(metrics["completed"])
                self.assertFalse(metrics["fell"])
                self.assertEqual(metrics["numerical_warnings"], 0)
                self.assertEqual(metrics["control_steps"], 500)
                self.assertEqual(metrics["physics_steps"], 10000)
                self.assertEqual(metrics["simulated_seconds"], 10)
                self.assertLess(metrics["joint_rmse_rad"], 0.25)
                self.assertLess(metrics["mean_body_position_error_m"], 0.06)
                self.assertGreater(metrics["min_root_height_m"], 0.7)
                self.assertLess(metrics["max_pelvis_tilt_deg"], 10)
                frames = case["frames"]
                self.assertEqual(len(frames), 251)
                np.testing.assert_allclose([frame["time"] for frame in frames], np.linspace(0, 10, 251))
                for field in ("body", "reference"):
                    points = np.asarray([frame[field] for frame in frames])
                    self.assertEqual(points.shape, (251, 33, 3))
                    self.assertTrue(np.isfinite(points).all())


if __name__ == "__main__":
    unittest.main()
