import unittest
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from constrained_ik import ConstrainedIK, PoseTargets, orientation_residual
from fetch_assets import MODEL_DIR
from rollout import load_model


class ConstrainedIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = Path(".research-cache/protomotions")
        directory = assets / MODEL_DIR
        meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        cls.default = np.asarray(config["robot"]["default_dof_pos"])
        cls.model = load_model(assets, meta, config)

    def setUp(self):
        self.solver = ConstrainedIK(self.model, self.default)

    def targets(self, q, root=(0., 0., .95)):
        data = mujoco.MjData(self.model)
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root, [1, 0, 0, 0], q
        mujoco.mj_forward(self.model, data)
        return PoseTargets(data.qpos[:3].copy(), data.qpos[3:7].copy(),
                           data.xpos[self.solver.bodies].copy(), data.xmat[self.solver.bodies].reshape(2, 3, 3).copy(),
                           data.xmat[self.solver.torso].reshape(3, 3).copy())

    def test_rotation_log_derivative_world_frame_and_near_pi(self):
        target = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
        for angle in (.00001, .7, 3.05):
            current = Rotation.from_rotvec(np.array([1, 2, 3]) / np.sqrt(14) * angle).as_matrix() @ target
            _, jacobian = orientation_residual(current, target)
            numeric = np.empty((3, 3))
            for i in range(3):
                delta = np.eye(3)[i] * 1e-6
                numeric[:, i] = (orientation_residual(Rotation.from_rotvec(delta).as_matrix() @ current, target)[0]
                                 - orientation_residual(Rotation.from_rotvec(-delta).as_matrix() @ current, target)[0]) / 2e-6
            np.testing.assert_allclose(jacobian, numeric, atol=1e-8)

    def test_actual_mujoco_pose_floor_and_tendon_jacobians(self):
        target = self.targets(self.default)
        x = np.r_[target.root, self.default[:15]] + np.linspace(-.03, .03, 18)
        _, analytic = self.solver.residual(x, target, self.default[:15])
        _, floor_jac = self.solver.floor(x, target)
        for i in range(18):
            delta = np.eye(18)[i] * 1e-6
            numerical = (self.solver.residual(x + delta, target, self.default[:15])[0]
                         - self.solver.residual(x - delta, target, self.default[:15])[0]) / 2e-6
            np.testing.assert_allclose(analytic[:, i], numerical, atol=1e-7)
            numerical_floor = (self.solver.floor(x + delta, target)[0] - self.solver.floor(x - delta, target)[0]) / 2e-6
            np.testing.assert_allclose(floor_jac[:, i], numerical_floor, atol=1e-8)
        self.solver.set_pose(x, target)
        np.testing.assert_allclose(self.solver.data.ten_length,
                                   self.solver.tendon_matrix @ x + self.solver.tendon_offset, atol=1e-12)

    def test_recovers_known_reachable_rocking_pose_trajectory(self):
        seed = None
        worst_position = worst_angle = 0.
        for phase in np.linspace(0, 2 * np.pi, 21):
            q = self.default.copy()
            q[[0, 3, 4, 6, 9, 10]] += np.sin(phase) * np.array([-.15, .25, .08, .15, -.20, -.08])
            q[[1, 5, 7, 11, 12, 13, 14]] += .03 * np.cos(phase)
            target = self.targets(q)
            lowest = self.solver.floor(np.r_[target.root, q[:15]], target)[0].min()
            target = self.targets(q, root=target.root - [0, 0, lowest - .002])
            previous = seed
            seed, fitted, diagnostics = self.solver.solve(target, seed, root_bound=np.zeros(3),
                                                        previous=previous, max_joint_step=.16)
            self.assertTrue(diagnostics["optimizer_success"], diagnostics)
            self.assertLess(diagnostics["tendon_max_violation"], 1e-8)
            self.assertGreater(diagnostics["floor_min_m"], .0019)
            if previous is not None:
                self.assertLessEqual(np.abs(seed[3:] - previous[3:]).max(), .16)
            worst_position = max(worst_position, max(diagnostics["foot_error_m"]))
            worst_angle = max(worst_angle, max(diagnostics["foot_angle_error_rad"]))
            np.testing.assert_allclose(fitted[7:22], q[:15], atol=1e-4)
        self.assertLess(worst_position, 1e-5)
        self.assertLess(worst_angle, 1e-5)

    def test_impossible_target_never_returns_unbounded_or_nan_pose(self):
        target = self.targets(self.default)
        target.feet[0, 0] += 2
        _, q, report = self.solver.solve(target, max_iterations=60)
        self.assertTrue(np.isfinite(q).all())
        self.assertGreater(max(report["foot_error_m"]), .5)
        self.assertTrue(np.all(q[7:] >= self.model.jnt_range[1:, 0]))
        self.assertTrue(np.all(q[7:] <= self.model.jnt_range[1:, 1]))
        self.assertLess(report["tendon_max_violation"], 1e-6)


if __name__ == "__main__":
    unittest.main()
