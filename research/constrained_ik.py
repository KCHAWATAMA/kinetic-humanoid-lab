"""G1 pose fitting with exact joint/tendon bounds and capsule-floor constraints."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize
from scipy.spatial.transform import Rotation

from locomotion import FEET


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def orientation_residual(current, target):
    error = Rotation.from_matrix(current @ target.T).as_rotvec()
    angle = np.linalg.norm(error)
    cross = skew(error)
    coefficient = 1 / 12 + angle * angle / 720 if angle < 1e-4 else (1 - .5 * angle / np.tan(.5 * angle)) / angle ** 2
    return error, np.eye(3) - .5 * cross + coefficient * cross @ cross


def foot_capsules(model, body):
    points, radii = [], []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        axis = rotation.reshape(3, 3)[:, 2]
        for sign in (-1, 1):
            points.append(model.geom_pos[geom] + sign * model.geom_size[geom, 1] * axis)
            radii.append(model.geom_size[geom, 0])
    if not points:
        raise ValueError("G1 foot capsules are missing.")
    return np.asarray(points), np.asarray(radii)


@dataclass
class PoseTargets:
    root: np.ndarray
    root_quaternion: np.ndarray
    feet: np.ndarray
    foot_rotations: np.ndarray
    torso_rotation: np.ndarray


class ConstrainedIK:
    def __init__(self, model, default):
        self.model, self.default = model, np.asarray(default)
        self.data = mujoco.MjData(model)
        self.dofs = np.r_[np.arange(3), np.arange(6, 21)]
        self.bodies = [model.body(name).id for name in FEET]
        self.torso = model.body("torso_link").id
        self.capsules = [foot_capsules(model, body) for body in self.bodies]
        self.data.qpos[7:] = default
        mujoco.mj_forward(model, self.data)
        self.tendon_matrix = self.data.ten_J.reshape(model.ntendon, model.nv)[:, self.dofs].copy()
        self.tendon_offset = self.data.ten_length - self.tendon_matrix @ np.r_[self.data.qpos[:3], default[:15]]
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def set_pose(self, x, target):
        self.data.qpos[:3] = x[:3]
        self.data.qpos[3:7] = target.root_quaternion
        self.data.qpos[7:] = self.default
        self.data.qpos[7:22] = x[3:]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_tendon(self.model, self.data)

    def residual(self, x, target, prior):
        self.set_pose(x, target)
        errors, jacobians = [], []
        for side, body in enumerate(self.bodies):
            mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, body)
            errors.append(10 * (self.data.xpos[body] - target.feet[side]))
            jacobians.append(10 * self.jacp[:, self.dofs])
            error, derivative = orientation_residual(self.data.xmat[body].reshape(3, 3), target.foot_rotations[side])
            errors.append(error)
            jacobians.append(derivative @ self.jacr[:, self.dofs])
        mujoco.mj_jacBody(self.model, self.data, None, self.jacr, self.torso)
        error, derivative = orientation_residual(self.data.xmat[self.torso].reshape(3, 3), target.torso_rotation)
        errors.extend([error, .1 * (x[:3] - target.root), .001 * (x[3:] - prior)])
        jacobians.extend([derivative @ self.jacr[:, self.dofs],
                          np.c_[np.eye(3) * .1, np.zeros((3, 15))],
                          np.c_[np.zeros((15, 3)), np.eye(15) * .001]])
        return np.concatenate(errors), np.vstack(jacobians)

    def floor(self, x, target):
        self.set_pose(x, target)
        heights, derivatives = [], []
        for body, (points, radii) in zip(self.bodies, self.capsules):
            world = self.data.xpos[body] + points @ self.data.xmat[body].reshape(3, 3).T
            for point, radius in zip(world, radii):
                mujoco.mj_jac(self.model, self.data, self.jacp, None, point, body)
                heights.append(point[2] - radius)
                derivatives.append(self.jacp[2, self.dofs].copy())
        return np.asarray(heights), np.asarray(derivatives)

    def solve(self, target, seed=None, root_bound=(.12, .12, .12), max_iterations=150, seed_evaluations=80,
              previous=None, max_joint_step=None):
        prior = previous[3:] if previous is not None else self.default[:15] if seed is None else seed[3:]
        lower = np.r_[target.root - root_bound, self.model.jnt_range[1:16, 0] + 1e-7]
        upper = np.r_[target.root + root_bound, self.model.jnt_range[1:16, 1] - 1e-7]
        if previous is not None and max_joint_step is not None:
            lower[3:] = np.maximum(lower[3:], previous[3:] - max_joint_step)
            upper[3:] = np.minimum(upper[3:], previous[3:] + max_joint_step)
        x0 = np.clip(np.r_[target.root, prior] if seed is None else seed, lower, upper)
        def objective(x):
            residual, jacobian = self.residual(x, target, prior)
            return .5 * residual @ residual, jacobian.T @ residual
        active = upper > lower
        def expand(values):
            x = x0.copy()
            x[active] = values
            return x
        initialization = least_squares(lambda values: self.residual(expand(values), target, prior)[0],
                                       x0[active], jac=lambda values: self.residual(expand(values), target, prior)[1][:, active],
                                       bounds=(lower[active], upper[active]), x_scale="jac", max_nfev=seed_evaluations,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
        x0 = expand(initialization.x)
        constraint = LinearConstraint(self.tendon_matrix, self.model.tendon_range[:, 0] - self.tendon_offset + 1e-7,
                                      self.model.tendon_range[:, 1] - self.tendon_offset - 1e-7)
        solution = minimize(objective, x0, jac=True, method="SLSQP", bounds=Bounds(lower, upper),
                            constraints=[constraint, {"type": "ineq", "fun": lambda x: self.floor(x, target)[0],
                                                       "jac": lambda x: self.floor(x, target)[1]}],
                            options={"ftol": 1e-12, "maxiter": max_iterations})
        self.set_pose(solution.x, target)
        foot_error = np.linalg.norm(self.data.xpos[self.bodies] - target.feet, axis=1)
        angle_error = [np.linalg.norm(orientation_residual(self.data.xmat[body].reshape(3, 3), target.foot_rotations[i])[0])
                       for i, body in enumerate(self.bodies)]
        torso_error = np.linalg.norm(orientation_residual(self.data.xmat[self.torso].reshape(3, 3), target.torso_rotation)[0])
        diagnostics = {
            "optimizer_success": bool(solution.success), "message": str(solution.message), "iterations": int(solution.nit),
            "bounded_ls_evaluations": int(initialization.nfev),
            "foot_error_m": foot_error.tolist(), "foot_angle_error_rad": angle_error,
            "torso_angle_error_rad": float(torso_error), "root_adjustment_m": (solution.x[:3] - target.root).tolist(),
            "floor_min_m": float(self.floor(solution.x, target)[0].min()),
            "tendon_max_violation": float(max(0., np.max(self.model.tendon_range[:, 0] - self.data.ten_length),
                                              np.max(self.data.ten_length - self.model.tendon_range[:, 1]))),
        }
        return solution.x, self.data.qpos.copy(), diagnostics
