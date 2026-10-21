"""Matched-controller mechanics and failure accounting, independent of who wins."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

from evaluate import PROTOCOL_PATH, aggregate, make_pushes
from fetch_assets import MODEL_DIR
from rollout import Push, controller_target, force_at_step, grounded_root_height, load_model, run_case, verify_assets


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = Path(".research-cache/protomotions")
        verify_assets(assets)
        directory = assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text(encoding="utf-8"))
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text(encoding="utf-8"))
        cls.default = np.asarray(config["robot"]["default_dof_pos"], dtype=np.float64)
        cls.model = load_model(assets, cls.meta, config)
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_controller_selection_keeps_direct_pd_independent_of_policy(self):
        reference = {"q": self.default.copy(), "v": np.zeros(29), "anchor": np.array([0, 0, 0, 1])}
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        previous = np.arange(29, dtype=np.float32)
        captured = []
        meta = self.meta

        class Policy:
            def run(self, names, feed):
                captured.append(feed)
                return [np.zeros((1, 29)), np.ones((1, 29)), np.array(meta["control"]["stiffness"]),
                        np.array(meta["control"]["damping"])]

        policy = Policy()
        target, latency = controller_target("direct_pd", policy, data, meta, reference, [reference] * 4, previous)
        self.assertEqual(captured, [])
        self.assertIsNone(latency)
        np.testing.assert_array_equal(target, reference["q"])
        target[0] += 1
        np.testing.assert_array_equal(reference["q"], self.default)
        target, latency = controller_target("onnx", policy, data, meta, reference, [reference] * 4, previous)
        np.testing.assert_array_equal(target, np.ones(29))
        np.testing.assert_array_equal(captured[0]["historical_processed_actions"][0, 0], previous)
        self.assertGreaterEqual(latency, 0)
        with self.assertRaises(ValueError):
            controller_target("unknown", policy, data, meta, reference, [], previous)
        with self.assertRaises(ValueError):
            controller_target("onnx", None, data, meta, reference, [], previous)

    def test_fixed_matrix_and_seeded_pushes(self):
        self.assertEqual(self.protocol["references"], ["stand", "arm_sweep"])
        self.assertEqual(self.protocol["controllers"], ["onnx", "direct_pd"])
        self.assertEqual([item["seed"] for item in self.protocol["conditions"]], [None, 17, 29])
        self.assertEqual(self.protocol["seconds_per_episode"], 5)
        self.assertEqual(make_pushes(self.protocol, None), ())
        pushes = make_pushes(self.protocol, 17)
        self.assertEqual(pushes, make_pushes(self.protocol, 17))
        self.assertNotEqual(pushes, make_pushes(self.protocol, 29))
        self.assertEqual([(p.start_step, p.end_step) for p in pushes], [(50, 60), (150, 160)])
        for push in pushes:
            self.assertAlmostEqual(np.linalg.norm(push.force_world_n), 60)
            self.assertEqual(push.force_world_n[2], 0)
            np.testing.assert_array_equal(force_at_step(pushes, push.start_step - 1), np.zeros(3))
            np.testing.assert_array_equal(force_at_step(pushes, push.start_step), push.force_world_n)
            np.testing.assert_array_equal(force_at_step(pushes, push.end_step - 1), push.force_world_n)
            np.testing.assert_array_equal(force_at_step(pushes, push.end_step), np.zeros(3))
            self.assertAlmostEqual((push.end_step - push.start_step) * 0.02 * 60, 12)

    def test_aggregation_includes_falls_and_partial_episodes(self):
        def episode(completed, fell, error, samples):
            return {"controller": "onnx", "metrics": {
                "completed": completed, "fell": fell, "joint_rmse_rad": error, "error_samples": samples,
                "mean_body_position_error_m": error, "simulated_seconds": (samples - 1) * 0.02, "wall_seconds": 1,
            }}
        episodes = [episode(True, False, 1, 251), episode(True, True, 3, 251), episode(False, False, 10, 11)]
        summary = aggregate(episodes, 3)[0]
        self.assertEqual((summary["successes"], summary["falls"], summary["failures"], summary["early_terminations"]), (1, 1, 2, 1))
        self.assertAlmostEqual(summary["failure_rate"], 2 / 3)
        self.assertAlmostEqual(summary["fall_rate"], 1 / 3)
        self.assertAlmostEqual(summary["observed_joint_rmse_rad"], math.sqrt((251 + 9 * 251 + 100 * 11) / 513))
        self.assertIsNone(summary["full_horizon_joint_rmse_rad"])
        self.assertIsNone(aggregate(episodes[:2], 3)[0]["full_horizon_joint_rmse_rad"])
        self.assertIsNotNone(aggregate(episodes[:2], 2)[0]["full_horizon_joint_rmse_rad"])
        self.assertEqual(aggregate([], 6)[0]["attempted_episodes"], 0)
        self.assertIsNone(aggregate([], 6)[0]["failure_rate"])

    def test_direct_pd_uses_same_physics_and_exact_force_schedule(self):
        seen = []
        mj_step = mujoco.mj_step

        def step(model, data):
            seen.append(data.xfrc_applied.copy())
            mj_step(model, data)

        with patch("rollout.mujoco.mj_step", side_effect=step), redirect_stdout(io.StringIO()):
            result = run_case(self.model, None, self.meta, self.default, "stand", 1, controller="direct_pd",
                              pushes=(Push(10, 20, (60.0, 0.0, 0.0)),), record_frames=False)
        metrics = result["metrics"]
        self.assertTrue(metrics["completed"])
        self.assertEqual((metrics["controller_calls"], metrics["control_steps"], metrics["physics_steps"]), (50, 50, 1000))
        self.assertEqual(metrics["error_samples"], 51)
        self.assertIsNone(metrics["mean_inference_ms"])
        self.assertEqual(result["frames"], [])
        self.assertAlmostEqual(metrics["initial_root_height_m"], grounded_root_height(self.model, self.default))
        forces = np.asarray(seen)
        expected = np.zeros_like(forces)
        expected[200:400, 1, 0] = 60
        np.testing.assert_array_equal(forces, expected)

    def test_nonfinite_policy_output_is_recorded_as_early_failure(self):
        class InvalidPolicy:
            def run(self, names, feed):
                return [np.full((1, 29), np.nan)] * 4

        with redirect_stdout(io.StringIO()):
            result = run_case(self.model, InvalidPolicy(), self.meta, self.default, "stand", 1, record_frames=False)
        self.assertFalse(result["metrics"]["completed"])
        self.assertEqual(result["metrics"]["controller_calls"], 1)
        self.assertEqual(result["metrics"]["control_steps"], 0)
        self.assertEqual(result["metrics"]["simulated_seconds"], 0)
        self.assertEqual(result["metrics"]["error_samples"], 1)
        self.assertIn("Non-finite", result["termination"])
        json.dumps(result, allow_nan=False)

    def test_wall_budget_keeps_partial_evidence(self):
        with patch("rollout.time.perf_counter", side_effect=[0, 91, 91]), redirect_stdout(io.StringIO()):
            result = run_case(self.model, None, self.meta, self.default, "stand", 5, controller="direct_pd",
                              record_frames=False, wall_limit_seconds=90)
        self.assertEqual(result["termination"], "wall_budget_exceeded")
        self.assertFalse(result["metrics"]["completed"])
        self.assertEqual(result["metrics"]["controller_calls"], 0)
        self.assertEqual(result["metrics"]["wall_seconds"], 91)
        json.dumps(result, allow_nan=False)

    def test_saved_matrix_preserves_every_pair_and_recomputable_summary(self):
        result = json.loads(Path("public/physics-comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["protocol"], self.protocol)
        self.assertEqual(result["provenance"], verify_assets(Path(".research-cache/protomotions")))
        self.assertEqual(len(result["episodes"]), 12)
        self.assertEqual(result["summary"], aggregate(result["episodes"], 6))
        self.assertEqual(len({episode["metrics"]["initial_root_height_m"] for episode in result["episodes"]}), 1)
        for reference in self.protocol["references"]:
            for condition in self.protocol["conditions"]:
                pair = [episode for episode in result["episodes"] if episode["name"] == reference and episode["condition"] == condition["name"]]
                self.assertEqual({episode["controller"] for episode in pair}, {"onnx", "direct_pd"})
                expected_pushes = json.loads(json.dumps([asdict(push) for push in make_pushes(self.protocol, condition["seed"])]))
                for episode in pair:
                    self.assertEqual(episode["pushes"], expected_pushes)
                    self.assertIsNone(episode["termination"])
                    metrics = episode["metrics"]
                    self.assertTrue(metrics["completed"])
                    self.assertEqual(metrics["simulated_seconds"], 5)
                    self.assertEqual(metrics["controller_calls"], 250)
                    self.assertEqual(metrics["control_steps"], 250)
                    self.assertEqual(metrics["physics_steps"], 5000)
                    self.assertEqual(metrics["error_samples"], 251)
                    self.assertEqual(metrics["numerical_warnings"], 0)
                    self.assertEqual(metrics["fell"], metrics["first_fall_time_s"] is not None)
                    for value in metrics.values():
                        if isinstance(value, (int, float)):
                            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
