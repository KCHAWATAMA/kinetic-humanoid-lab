"""Bounded self-authored G1 locomotion experiment with physical contact metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import mujoco
import numpy as np
import yaml

from fetch_assets import MODEL_DIR
from rollout import grounded_root_height, load_model, make_session, run_case, verify_assets, xyzw

PROTOCOL_PATH = Path(__file__).with_name("locomotion-protocol.json")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class ReferenceGenerationError(ValueError):
    pass


def smooth(value: float) -> float:
    u = float(np.clip(value, 0, 1))
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def walk_targets(t: float, initial_feet: np.ndarray, height: float, protocol: dict) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    feet = initial_feet.copy()
    root = np.array([0.0, 0.0, height])
    stance = [True, True]
    for step in range(protocol["step_count"]):
        start = protocol["initial_hold_s"] + step * protocol["step_period_s"]
        local = t - start
        if local <= 0:
            break
        side = step % 2
        old_midpoint = float(np.mean(feet[:, 0] - initial_feet[:, 0]))
        end_x = initial_feet[side, 0] + (step + 1) * protocol["step_length_m"]
        swing_start, swing_end = protocol["swing_start_s"], protocol["swing_end_s"]
        progress = smooth((local - swing_start) / (swing_end - swing_start))
        feet[side, 0] += (end_x - feet[side, 0]) * progress
        root[0] = float(np.mean(feet[:, 0] - initial_feet[:, 0]))
        if local < protocol["step_period_s"]:
            phase = float(np.clip((local - swing_start) / (swing_end - swing_start), 0, 1))
            feet[side, 2] += protocol["foot_clearance_m"] * 16 * phase ** 2 * (1 - phase) ** 2
            shift = smooth(local / swing_start) * (1 - smooth((local - swing_end) / (protocol["step_period_s"] - swing_end)))
            root[1] = (-1 if side == 0 else 1) * protocol["stance_root_shift_m"] * shift
            stance[side] = not (swing_start < local < swing_end)
            if local <= swing_start:
                root[0] = old_midpoint
            break
    return root, feet, stance


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return sum((np.cross(current[:, i], target[:, i]) for i in range(3)), np.zeros(3)) * 0.5


def solve_foot(model: mujoco.MjModel, data: mujoco.MjData, body: int, joint_ids: list[int],
               target: np.ndarray, rotation: np.ndarray) -> tuple[float, float]:
    q_indices = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(40):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        position_error = target - data.xpos[body]
        orient_error = rotation_error(data.xmat[body].reshape(3, 3), rotation)
        if np.linalg.norm(position_error) < 1e-6 and np.linalg.norm(orient_error) < 1e-5:
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, body)
        jacobian = np.vstack((jacp[:, dofs], jacr[:, dofs]))
        error = np.concatenate((position_error, orient_error))
        update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1e-6, error)
        data.qpos[q_indices] = np.clip(data.qpos[q_indices] + np.clip(update, -0.08, 0.08), limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)
    mujoco.mj_kinematics(model, data)
    relative = rotation @ data.xmat[body].reshape(3, 3).T
    angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(target - data.xpos[body])), angle


def generate_reference(model: mujoco.MjModel, meta: dict, default: np.ndarray, protocol: dict) -> tuple[list[dict], dict]:
    dt = meta["timing"]["control_dt"]
    count = round(protocol["seconds"] / dt) + max(meta["motion"]["future_step_indices"]) + 1
    data = mujoco.MjData(model)
    height = grounded_root_height(model, default)
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = [0, 0, height], [1, 0, 0, 0], default
    mujoco.mj_forward(model, data)
    bodies = [model.body(name).id for name in FEET]
    joints = [[model.joint(name).id for name in meta["robot"]["joint_names"][offset:offset + 6]] for offset in (0, 6)]
    initial_feet = data.xpos[bodies].copy()
    initial_rotations = data.xmat[bodies].reshape(2, 3, 3).copy()
    frames = []
    max_position_error = max_angle_error = 0.0
    for i in range(count):
        root, targets, stance = walk_targets(i * dt, initial_feet, height, protocol)
        data.qpos[:3] = root
        for side in range(2):
            position_error, angle_error = solve_foot(model, data, bodies[side], joints[side], targets[side], initial_rotations[side])
            max_position_error = max(max_position_error, position_error)
            max_angle_error = max(max_angle_error, angle_error)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_tendon(model, data)
        if np.any(data.qpos[7:] < model.jnt_range[1:, 0]) or np.any(data.qpos[7:] > model.jnt_range[1:, 1]):
            raise ReferenceGenerationError(f"Reference joint limit violation at {i * dt:.2f} s")
        if np.any(data.ten_length < model.tendon_range[:, 0] - 1e-6) or np.any(data.ten_length > model.tendon_range[:, 1] + 1e-6):
            raise ReferenceGenerationError(f"Reference ankle tendon limit violation at {i * dt:.2f} s")
        frames.append({
            "q": data.qpos[7:].copy(), "anchor": xyzw(data.xquat[meta["robot"]["anchor_body_index"] + 1]),
            "body": data.xpos[1:].copy(), "root": root.copy(), "feet": targets.copy(), "stance": stance,
        })
    positions = np.stack([frame["q"] for frame in frames])
    velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    for frame, velocity in zip(frames, velocities):
        frame["v"] = velocity.astype(np.float32)
        frame["q"] = frame["q"].astype(np.float32)
    diagnostics = {
        "frames": count, "max_foot_position_error_m": max_position_error, "max_foot_orientation_error_rad": max_angle_error,
        "max_joint_velocity_rad_s": float(np.max(np.abs(velocities))),
        "target_forward_progress_m": float(frames[round(protocol["seconds"] / dt)]["root"][0]),
        "joint_limits_satisfied": True, "ankle_tendon_limits_satisfied": True,
    }
    gates = protocol["reference_gates"]
    if max_position_error > gates["maximum_foot_position_error_m"] or max_angle_error > gates["maximum_foot_orientation_error_rad"] or diagnostics["max_joint_velocity_rad_s"] > gates["maximum_joint_velocity_rad_s"]:
        raise ReferenceGenerationError(f"Reference quality gates failed: {diagnostics}")
    return frames, diagnostics


def contact_telemetry(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    normal = np.zeros(2)
    weighted_speed = np.zeros(2)
    bodies = [model.body(name).id for name in FEET]
    floor = model.geom("floor").id
    wrench = np.zeros(6)
    jacp = np.zeros((3, model.nv))
    for index in range(data.ncon):
        contact = data.contact[index]
        geom = contact.geom2 if contact.geom1 == floor else contact.geom1 if contact.geom2 == floor else -1
        if geom < 0 or model.geom_bodyid[geom] not in bodies:
            continue
        side = bodies.index(model.geom_bodyid[geom])
        mujoco.mj_contactForce(model, data, index, wrench)
        force = max(0.0, float(wrench[0]))
        mujoco.mj_jac(model, data, jacp, None, contact.pos, bodies[side])
        speed = float(np.linalg.norm((jacp @ data.qvel)[:2]))
        normal[side] += force
        weighted_speed[side] += force * speed
    speeds = np.divide(weighted_speed, normal, out=np.zeros(2), where=normal > 0)
    return normal, speeds


class FootStepCounter:
    def __init__(self, definition: dict):
        self.definition = definition
        self.last_loaded: tuple[float, np.ndarray] | None = None
        self.liftoff: tuple[float, np.ndarray] | None = None
        self.peak_z = -math.inf
        self.steps = 0

    def sample(self, t: float, position: np.ndarray, loaded: bool) -> None:
        if loaded:
            if self.liftoff is not None:
                start, origin = self.liftoff
                if (t - start >= self.definition["minimum_air_time_s"]
                    and self.peak_z - origin[2] >= self.definition["minimum_clearance_above_liftoff_m"]
                    and position[0] - origin[0] >= self.definition["minimum_forward_landing_distance_m"]):
                    self.steps += 1
            self.last_loaded = (t, position.copy())
            self.liftoff = None
            self.peak_z = -math.inf
        elif self.last_loaded is not None:
            if self.liftoff is None:
                self.liftoff = (t, self.last_loaded[1].copy())
            self.peak_z = max(self.peak_z, float(position[2]))


class LocomotionObserver:
    def __init__(self, protocol: dict, dt: float):
        self.protocol, self.dt = protocol, dt
        self.samples: list[dict] = []
        self.counters = [FootStepCounter(protocol["step_detection"]) for _ in FEET]

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, step: int, reference: dict):
        normal, speeds = contact_telemetry(model, data)
        loaded = normal >= self.protocol["step_detection"]["loaded_foot_normal_force_n"]
        feet = data.xpos[[model.body(name).id for name in FEET]]
        for side in range(2):
            self.counters[side].sample(step * self.dt, feet[side], bool(loaded[side]))
        self.samples.append({
            "time": round(step * self.dt, 6), "root": data.qpos[:3].tolist(), "referenceRoot": reference["root"].tolist(),
            "feet": feet.tolist(), "normalForceN": normal.tolist(), "loaded": loaded.tolist(),
            "contactSlipMS": speeds.tolist(), "referenceStance": reference["stance"],
        })

    def summarize(self, physics: dict) -> dict:
        roots = np.asarray([sample["root"] for sample in self.samples])
        targets = np.asarray([sample["referenceRoot"] for sample in self.samples])
        slip_by_foot = [[sample["contactSlipMS"][side] for sample in self.samples if sample["loaded"][side]] for side in range(2)]
        support = [any(sample["loaded"]) for sample in self.samples if sample["time"] >= 0.5]
        mean_slip = [float(np.mean(values)) if values else None for values in slip_by_foot]
        p95_slip = [float(np.percentile(values, 95)) if values else None for values in slip_by_foot]
        metrics = {
            "forward_progress_m": float(roots[-1, 0] - roots[0, 0]), "target_forward_progress_m": float(targets[-1, 0] - targets[0, 0]),
            "final_lateral_drift_m": float(roots[-1, 1] - roots[0, 1]),
            "root_xy_rmse_m": float(np.sqrt(np.mean(np.sum((roots[:, :2] - targets[:, :2]) ** 2, axis=1)))),
            "verified_steps_per_foot": [counter.steps for counter in self.counters],
            "support_fraction_after_half_second": float(np.mean(support)) if support else 0.0,
            "loaded_samples_per_foot": [len(values) for values in slip_by_foot],
            "mean_contact_slip_m_s": mean_slip, "p95_contact_slip_m_s": p95_slip,
            "integrated_sampled_contact_slip_m": [float(sum(values) * self.dt) for values in slip_by_foot],
        }
        c = self.protocol["criteria"]
        gates = {
            "full_duration": physics["completed"] and physics["simulated_seconds"] == self.protocol["seconds"],
            "no_fall": not physics["fell"], "no_numerical_warning": physics["numerical_warnings"] == 0,
            "forward_progress": c["minimum_forward_progress_m"] <= metrics["forward_progress_m"] <= c["maximum_forward_progress_m"],
            "lateral_drift": abs(metrics["final_lateral_drift_m"]) <= c["maximum_final_lateral_drift_m"],
            "root_tracking": metrics["root_xy_rmse_m"] <= c["maximum_root_xy_rmse_m"],
            "joint_tracking": physics["joint_rmse_rad"] <= c["maximum_joint_rmse_rad"],
            "body_tracking": physics["mean_body_position_error_m"] <= c["maximum_mean_body_error_m"],
            "verified_steps": all(value >= c["minimum_verified_steps_per_foot"] for value in metrics["verified_steps_per_foot"]),
            "supported_motion": metrics["support_fraction_after_half_second"] >= c["minimum_support_fraction_after_half_second"],
            "mean_contact_slip": all(value is not None and value <= c["maximum_mean_contact_slip_m_s"] for value in mean_slip),
            "p95_contact_slip": all(value is not None and value <= c["maximum_p95_contact_slip_m_s"] for value in p95_slip),
        }
        return {"passed": all(gates.values()), "gates": gates, "metrics": metrics, "samples": self.samples}


def run_attempt(assets: Path, output_dir: Path, attempt: int) -> dict:
    if attempt not in (1, 2):
        raise ValueError("This bounded experiment permits only attempts 1 and 2.")
    protocol_path = PROTOCOL_PATH if attempt == 1 else PROTOCOL_PATH.with_name("locomotion-protocol-02.json")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if not 1 <= attempt <= protocol["maximum_attempts"] or protocol["seconds"] != 8:
        raise ValueError("This bounded experiment permits attempts 1-2, eight seconds each.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"attempt-{attempt:02d}.json"
    result = {
        "schemaVersion": 1, "kind": "physical-locomotion-experiment", "attempt": attempt, "status": "running",
        "createdAt": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "protocolSha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "sourceSha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("locomotion.py", "rollout.py")},
        "provenance": verify_assets(assets),
        "contract": "ONNX consumes future joints/velocities and torso rotations, not root position or linear velocity. Root trajectory is used only for reference IK and error measurement.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "onnxruntime")},
                    "provider": "CPUExecutionProvider", "inference_threads": 1},
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    directory = assets / MODEL_DIR
    meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text(encoding="utf-8"))
    default = np.asarray(config["robot"]["default_dof_pos"])
    model = load_model(assets, meta, config)
    try:
        references, diagnostics = generate_reference(model, meta, default, protocol)
    except ReferenceGenerationError as error:
        result.update(status="reference_failed", error=str(error))
        save()
        raise
    result["referenceDiagnostics"] = diagnostics
    result["bodyNames"] = meta["robot"]["body_names"]
    result["parents"] = (model.body_parentid[1:] - 1).tolist()
    save()
    print(json.dumps({"attempt": attempt, "protocol": result["protocolSha256"], "reference": diagnostics}), flush=True)
    session = make_session(directory / "compiled_models/unified_pipeline.onnx", meta)
    observer = LocomotionObserver(protocol, meta["timing"]["control_dt"])
    physics = run_case(model, session, meta, default, "slow_walk", protocol["seconds"], reference_frames=references,
                       observe=observer, wall_limit_seconds=protocol["wall_limit_seconds"])
    result.update(status="complete", physics=physics, locomotion=observer.summarize(physics["metrics"]),
                  finishedAt=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps({key: value for key, value in result["locomotion"].items() if key != "samples"}, indent=2), flush=True)
    return result


def summarize_attempts(output_dir: Path) -> dict:
    attempts = []
    for number in (1, 2):
        path = output_dir / f"attempt-{number:02d}.json"
        if not path.exists():
            continue
        payload = path.read_bytes()
        result = json.loads(payload)
        item = {
            "attempt": number, "status": result["status"], "artifact": path.name,
            "artifactSha256": hashlib.sha256(payload).hexdigest(),
            "protocol": result["protocol"], "protocolSha256": result["protocolSha256"],
        }
        if result["status"] == "complete":
            samples = result["locomotion"]["samples"]
            after_hold = [sample for sample in samples if sample["time"] >= result["protocol"]["initial_hold_s"]]
            feet = np.asarray([sample["feet"] for sample in after_hold])
            swing_loads = [[sample["normalForceN"][side] for sample in after_hold if not sample["referenceStance"][side]] for side in (0, 1)]
            item.update(
                passed=result["locomotion"]["passed"], gates=result["locomotion"]["gates"],
                metrics=result["locomotion"]["metrics"], physicsMetrics=result["physics"]["metrics"],
                referenceDiagnostics=result["referenceDiagnostics"],
                footHeightExcursionM=np.ptp(feet[:, :, 2], axis=0).tolist() if len(feet) else None,
                minimumSwingReferenceLoadN=[min(loads) if loads else None for loads in swing_loads],
            )
        elif result["status"] == "reference_failed":
            item.update(passed=False, error=result["error"])
        attempts.append(item)
    summary = {
        "schemaVersion": 1, "kind": "recorded-locomotion-attempts",
        "status": "complete" if len(attempts) == 2 and all(item["status"] in ("complete", "reference_failed") for item in attempts) else "incomplete",
        "attempts": attempts,
        "walkingAchieved": any(item.get("passed") is True for item in attempts),
        "interpretation": "Fixed local gates, not a benchmark or SOTA. Include both attempts; absence of falls and low slip alone are not walking.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--output-dir", type=Path, default=Path("public/locomotion"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--summarize", action="store_true", help="Rebuild the small browser summary without rerunning physics.")
    args = parser.parse_args()
    if args.summarize:
        summary = summarize_attempts(args.output_dir)
        print(json.dumps({"status": summary["status"], "attempts": len(summary["attempts"]), "walkingAchieved": summary["walkingAchieved"]}))
    else:
        result = run_attempt(args.assets, args.output_dir, args.attempt)
        summarize_attempts(args.output_dir)
        if not result["locomotion"]["passed"]:
            raise SystemExit("Recorded physics attempt did not meet locomotion criteria; preserve and inspect its failed gates.")
