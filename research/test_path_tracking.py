import json
from pathlib import Path
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from diagnose import expected_observation
from fetch_assets import MODEL_DIR
from path_tracking import HeadingFeedback, PROTOCOL, PolicyTrace, path_basis, reference_frames, relative_metrics, transform_state
from rollout import load_model, make_inputs, make_session


class PathTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL.read_text())
        cls.old = json.loads(Path(cls.protocol["reference_artifact"]).read_text())
        assets = Path(".research-cache/protomotions")
        directory = assets / MODEL_DIR
        cls.meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        cls.model = load_model(assets, cls.meta, config)
        cls.frames = reference_frames(cls.model, cls.meta, cls.old["referenceEvidence"])
        cls.session = make_session(directory / "compiled_models/unified_pipeline.onnx", cls.meta)

    def test_relative_metrics_ignore_common_world_origin_and_heading(self):
        targets = np.array([[0., 0., 1.], [1., .1, 1.], [2., 0., 1.]])
        roots = targets + [[0, 0, 0], [.1, -.2, 0], [.2, -.3, 0]]
        first = relative_metrics(roots, targets)
        turn = Rotation.from_euler("z", 1.3)
        shift = [10., -4., 0.]
        transformed = relative_metrics(turn.apply(roots) + shift, turn.apply(targets) + shift)
        for name, value in first.items():
            if name != "reference_world_y_displacement_m":
                self.assertAlmostEqual(value, transformed[name], places=12)
        self.assertAlmostEqual(first["final_cross_track_error_m"], -.3)
        self.assertAlmostEqual(first["final_along_track_error_m"], .2)
        samples = self.old["locomotion"]["samples"]
        actual = relative_metrics([s["root"] for s in samples], [s["referenceRoot"] for s in samples])
        self.assertAlmostEqual(actual["root_xy_rmse_m"], self.old["locomotion"]["metrics"]["root_xy_rmse_m"])
        self.assertGreater(abs(actual["final_cross_track_error_m"]), .7)

    def test_freejoint_world_translation_and_body_local_rotation_are_independent(self):
        q, v = self.frames[0]["qpos"].copy(), self.frames[0]["qvel"].copy()
        turn = Rotation.from_euler("z", .7)
        shift = np.array([1.2, -.7, 0])
        tq, tv = transform_state(q, v, turn, shift)
        np.testing.assert_array_equal(tv[3:], v[3:])
        np.testing.assert_allclose(tv[:3], turn.apply(v[:3]), atol=1e-14)
        future = q.copy()
        mujoco.mj_integratePos(self.model, future, v, 1e-6)
        transformed_future, _ = transform_state(future, v, turn, shift)
        integrated = tq.copy()
        mujoco.mj_integratePos(self.model, integrated, tv, 1e-6)
        np.testing.assert_allclose(integrated, transformed_future, atol=1e-14)
        np.testing.assert_allclose((future[:3] - q[:3]) / 1e-6, v[:3], atol=1e-10)
        local_angular = (Rotation.from_quat(q[[4, 5, 6, 3]]).inv()
                         * Rotation.from_quat(future[[4, 5, 6, 3]])).as_rotvec() / 1e-6
        np.testing.assert_allclose(local_angular, v[3:6], atol=1e-9)

    def test_actual_graph_is_yaw_invariant_and_accepts_only_the_intended_correction(self):
        data, moved = mujoco.MjData(self.model), mujoco.MjData(self.model)
        frame = self.frames[25]
        data.qpos[:], data.qvel[:] = frame["qpos"], frame["qvel"]
        turn = Rotation.from_euler("z", .7)
        moved.qpos[:], moved.qvel[:] = transform_state(data.qpos.copy(), data.qvel.copy(), turn, np.array([1., -2., 0.]))
        mujoco.mj_forward(self.model, data)
        mujoco.mj_forward(self.model, moved)
        transformed = reference_frames(self.model, self.meta, self.old["referenceEvidence"], turn, np.array([1., -2., 0.]))
        offsets = self.meta["motion"]["future_step_indices"]
        self.assertEqual(offsets, [1, 2, 4, 8])
        a = make_inputs(data, self.meta, [self.frames[25 + i] for i in offsets], np.zeros(29))
        b = make_inputs(moved, self.meta, [transformed[25 + i] for i in offsets], np.zeros(29))
        np.testing.assert_allclose(expected_observation(a), expected_observation(b), atol=2e-6)
        target = self.session.run(["joint_pos_targets"], a)[0]
        np.testing.assert_allclose(target, self.session.run(["joint_pos_targets"], b)[0], atol=1e-5)
        changed = HeadingFeedback.apply(a, .1)
        for name in a:
            if name != "mimic_future_anchor_rot":
                np.testing.assert_array_equal(changed[name], a[name])
        rotations = [Rotation.from_quat(feed["mimic_future_anchor_rot"][0]) for feed in (a, changed)]
        for i in range(4):
            np.testing.assert_allclose((rotations[0][0].inv() * rotations[0][i]).as_matrix(),
                                       (rotations[1][0].inv() * rotations[1][i]).as_matrix(), atol=2e-7)
        self.assertGreater(float(np.max(np.abs(self.session.run(["joint_pos_targets"], changed)[0] - target))), 1e-4)
        self.assertIs(HeadingFeedback.apply(a, 0.), a)

    def test_feedback_is_bounded_smooth_transform_invariant_and_does_not_mutate_physics(self):
        controller = HeadingFeedback(self.protocol["feedback"], self.frames, 186, .02)
        frame = self.frames[10]
        zero = controller.update(frame["root"], frame["qvel"][:3], frame)
        self.assertEqual(zero["angle_rad"], 0)
        along, cross = path_basis(self.frames, 186)
        root = frame["root"].copy()
        root[:2] -= cross
        angle = 0.
        for _ in range(100):
            result = controller.update(root, frame["qvel"][:3], frame)
            self.assertLessEqual(abs(result["angle_rad"]), np.radians(25) + 1e-12)
            self.assertLessEqual(abs(result["angle_rad"] - angle), np.radians(60) * .02 + 1e-12)
            angle = result["angle_rad"]
        self.assertGreater(angle, 0)
        turn, shift = Rotation.from_euler("z", .7), np.array([1., -2., 0.])
        moved = reference_frames(self.model, self.meta, self.old["referenceEvidence"], turn, shift)
        a = HeadingFeedback(self.protocol["feedback"], self.frames, 186, .02)
        b = HeadingFeedback(self.protocol["feedback"], moved, 186, .02)
        self.assertAlmostEqual(a.update(root, frame["qvel"][:3], frame)["angle_rad"],
                               b.update(turn.apply(root) + shift, turn.apply(frame["qvel"][:3]), moved[10])["angle_rad"])
        data = mujoco.MjData(self.model)
        data.qpos[:], data.qvel[:] = self.frames[0]["qpos"], self.frames[0]["qvel"]
        mujoco.mj_forward(self.model, data)
        before = data.qpos.copy(), data.qvel.copy()
        adapter = PolicyTrace(self.session, self.frames, a)
        adapter.state = 0, data.qpos[:3].copy(), data.qvel[:3].copy()
        feed = make_inputs(data, self.meta, [self.frames[i] for i in (1, 2, 4, 8)], np.zeros(29))
        adapter.run(["joint_pos_targets"], feed)
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        with self.assertRaisesRegex(ValueError, "causal"):
            adapter.run(["joint_pos_targets"], feed)


if __name__ == "__main__":
    unittest.main()
