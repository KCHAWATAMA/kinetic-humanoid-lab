"""One bounded CMU-to-G1 reference candidate; no policy-driven parameter tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from cmu import CACHE, forward, load
from fetch_assets import MODEL_DIR
from locomotion import FEET, rotation_error, smooth, solve_foot
from rollout import load_model, verify_assets, xyzw

PROTOCOL = Path(__file__).with_name("cmu-protocol.json")


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def contacts(mask: np.ndarray, fps: int, maximum_gap_s: float, minimum_stance_s: float) -> np.ndarray:
    result = mask.copy()
    for start, end in spans(~mask):
        if start > 0 and end < len(mask) and (end - start) / fps <= maximum_gap_s:
            result[start:end] = True
    for start, end in spans(result):
        if (end - start) / fps < minimum_stance_s:
            result[start:end] = False
    return result


def filtered(values: np.ndarray) -> np.ndarray:
    weights = np.array([1, 6, 15, 20, 15, 6, 1]) / 64
    padded = np.pad(values, ((3, 3), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, i], weights, mode="valid") for i in range(values.shape[1])], axis=1)


def quaternion(matrix: np.ndarray) -> np.ndarray:
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.ascontiguousarray(matrix).reshape(9))
    return result


def slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > 0.9995:
        result = a + fraction * (b - a)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)


def lock_feet(raw: np.ndarray, yaw: np.ndarray, stance: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray, float]:
    intervals = spans(stance)
    if not intervals:
        raise ValueError("No source stance interval for this foot.")
    corrected, corrected_yaw = raw.copy(), yaw.copy()
    anchors = [(np.array([*np.median(raw[a:b, :2], axis=0), height]), float(np.median(yaw[a:b])))
               for a, b in intervals]
    for (start, end), (anchor, heading) in zip(intervals, anchors):
        corrected[start:end], corrected_yaw[start:end] = anchor, heading
    gaps = [(0, intervals[0][0], None, 0)]
    gaps += [(intervals[i][1], intervals[i + 1][0], i, i + 1) for i in range(len(intervals) - 1)]
    gaps += [(intervals[-1][1], len(raw), len(intervals) - 1, None)]
    for start, end, previous, following in gaps:
        before = intervals[previous][1] - 1 if previous is not None else intervals[following][0]
        after = intervals[following][0] if following is not None else before
        left = anchors[previous if previous is not None else following]
        right = anchors[following if following is not None else previous]
        d0, d1 = left[0] - raw[before], right[0] - raw[after]
        for i in range(start, end):
            u = smooth((i - before) / (after - before)) if after > before else 0
            corrected[i] = raw[i] + (1 - u) * d0 + u * d1
            corrected[i, 2] = max(height, corrected[i, 2])
            corrected_yaw[i] = (1 - u) * left[1] + u * right[1]
    return corrected, corrected_yaw, float(np.max(np.linalg.norm(corrected - raw, axis=1)))


def torso_ik(model, data, target: np.ndarray) -> float:
    body = model.body("torso_link").id
    indices = np.arange(12, 15)
    jacr = np.zeros((3, model.nv))
    for _ in range(30):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        error = rotation_error(data.xmat[body].reshape(3, 3), target)
        if np.linalg.norm(error) < 1e-6:
            break
        mujoco.mj_jacBody(model, data, None, jacr, body)
        jac = jacr[:, indices + 6]
        update = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(3) * 1e-6, error)
        data.qpos[indices + 7] = np.clip(data.qpos[indices + 7] + np.clip(update, -.08, .08),
                                         model.jnt_range[indices + 1, 0], model.jnt_range[indices + 1, 1])
    mujoco.mj_kinematics(model, data)
    return float(np.arccos(np.clip((np.trace(target @ data.xmat[body].reshape(3, 3).T) - 1) / 2, -1, 1)))


def generate(model, meta, default, protocol, cache=CACHE) -> tuple[list[dict], dict, dict]:
    skeleton, source, provenance = load(cache)
    fps, dt = protocol["source_fps"], meta["timing"]["control_dt"]
    if fps != 120 or dt != protocol["control_dt"] or protocol["retarget"]["source_smoothing_frames"] != 7:
        raise ValueError("Unsupported source/control/filter contract.")
    if protocol["seconds"] > (len(source) - 1) / fps or protocol["source_start_s"] != 0:
        raise ValueError("Selected source interval is unavailable.")
    poses = [forward(skeleton, frame) for frame in source]
    positions = {name: filtered(np.asarray([p[name] for p, _ in poses])) for name in skeleton.bones.keys() | {"root"}}
    transform = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
    displacement = transform @ (positions["root"][-1] - positions["root"][0])
    heading = np.arctan2(displacement[1], displacement[0])
    alignment = np.array([[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    world = alignment @ transform
    positions = {name: points @ world.T for name, points in positions.items()}
    origin = positions["root"][0] * [1, 1, 0]
    positions = {name: points - origin for name, points in positions.items()}
    rotations = {name: np.asarray([world @ r[name] @ transform.T for _, r in poses]) for name in ("root", "thorax")}
    settings = protocol["retarget"]
    stance_masks = []
    for side in ("l", "r"):
        sole = (positions[side + "foot"] + positions[side + "toes"]) / 2
        speed = np.linalg.norm(np.gradient(sole[:, :2], 1 / fps, axis=0), axis=1)
        mask = ((sole[:, 2] < np.percentile(sole[:, 2], 5) + settings["source_contact_height_above_fifth_percentile_m"])
                & (speed < settings["source_contact_speed_m_s"]))
        stance_masks.append(contacts(mask, fps, settings["maximum_contact_gap_s"], settings["minimum_source_stance_s"]))
    stance = np.stack(stance_masks, axis=1)
    data = mujoco.MjData(model)
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = 0, [1, 0, 0, 0], default
    mujoco.mj_forward(model, data)
    hip_offsets = np.asarray([data.xpos[model.body(side + "_hip_roll_link").id].copy() for side in ("left", "right")])
    lengths = []
    for side in ("left", "right"):
        hip, knee, foot = [data.xpos[model.body(side + suffix).id] for suffix in ("_hip_roll_link", "_knee_link", "_ankle_roll_link")]
        lengths.append(float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - foot)))
    source_lengths = [sum(skeleton.bones[side + part].length * skeleton.scale for part in ("femur", "tibia")) for side in ("l", "r")]
    scale = float(np.mean(np.asarray(lengths) / source_lengths))
    root = positions["root"] * scale
    raw_feet = np.stack([root + rotations["root"] @ hip_offsets[i]
                         + scale * (positions[side + "tibia"] - positions[side + "hipjoint"])
                         for i, side in enumerate(("l", "r"))], axis=1)
    if not stance.any(axis=0).all():
        raise ValueError("Both source feet require stance observations.")
    ground_offset = settings["foot_origin_height_m"] - float(np.median(raw_feet[:, :, 2][stance]))
    root[:, 2] += ground_offset
    raw_feet[:, :, 2] += ground_offset
    feet, yaws, corrections = [], [], []
    for i, side in enumerate(("l", "r")):
        direction = positions[side + "toes"] - positions[side + "tibia"]
        yaw = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
        foot, foot_yaw, correction = lock_feet(raw_feet[:, i], yaw, stance[:, i], settings["foot_origin_height_m"])
        feet.append(foot)
        yaws.append(foot_yaw)
        corrections.append(correction)
    feet, yaws = np.stack(feet, axis=1), np.stack(yaws, axis=1)
    source_time = np.arange(len(source)) / fps
    count = round(protocol["seconds"] / dt) + max(meta["motion"]["future_step_indices"]) + 1
    times = np.minimum(np.arange(count) * dt, protocol["seconds"])
    def interp(array):
        shape = array.shape[1:]
        flat = array.reshape(len(array), -1)
        return np.stack([np.interp(times, source_time, flat[:, i]) for i in range(flat.shape[1])], axis=1).reshape((count, *shape))
    roots, targets, headings = interp(root), interp(feet), interp(yaws)
    quaternions = {}
    for name in ("root", "thorax"):
        qs = [quaternion(matrix) for matrix in rotations[name]]
        quaternions[name] = np.asarray([slerp(qs[int(t * fps)], qs[min(int(t * fps) + 1, len(qs) - 1)], t * fps % 1) for t in times])
    sample_stance = np.asarray([stance[int(t * fps)] & stance[min(int(np.ceil(t * fps)), len(stance) - 1)] for t in times])
    frames, residuals, torso_residuals, lowerings, penetrations, tendon_violations = [], [], [], [], [], []
    body_ids = [model.body(name).id for name in FEET]
    joint_ids = [list(range(1, 7)), list(range(7, 13))]
    for i in range(count):
        rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(rotation_flat, quaternions["root"][i])
        root_rotation = rotation_flat.reshape(3, 3)
        rotated_hips = (root_rotation @ hip_offsets.T).T
        horizontal = np.linalg.norm(roots[i, :2] + rotated_hips[:, :2] - targets[i, :, :2], axis=1)
        reach = np.asarray(lengths) * settings["leg_reach_fraction"]
        if np.any(horizontal >= reach):
            raise ValueError(f"Unreachable horizontal foot target at {i * dt:.2f}s.")
        ceilings = targets[i, :, 2] - rotated_hips[:, 2] + np.sqrt(reach ** 2 - horizontal ** 2)
        lowering = max(0., float(roots[i, 2] - min(ceilings)))
        lowerings.append(lowering)
        roots[i, 2] -= min(lowering, settings["maximum_root_reach_lowering_m"])
        data.qpos[:3], data.qpos[3:7] = roots[i], quaternions["root"][i]
        mujoco.mju_quat2Mat(rotation_flat, quaternions["thorax"][i])
        torso_residuals.append(torso_ik(model, data, rotation_flat.reshape(3, 3)))
        for side in range(2):
            c, s = np.cos(headings[i, side]), np.sin(headings[i, side])
            target_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            residuals.append(solve_foot(model, data, body_ids[side], joint_ids[side], targets[i, side], target_rotation))
        mujoco.mj_forward(model, data)
        tendon_violations.append(bool(np.any(data.ten_length < model.tendon_range[:, 0] - 1e-6)
                                      or np.any(data.ten_length > model.tendon_range[:, 1] + 1e-6)))
        bottoms = []
        for geom in range(model.ngeom):
            if "foot" in (model.geom(geom).name or "") and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                bottoms.append(data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
                               - model.geom_size[geom, 1] * abs(data.geom_xmat[geom, 8]))
        penetrations.append(float(max(0., -min(bottoms))))
        frames.append({"q": data.qpos[7:].astype(np.float32).copy(), "root": roots[i].copy(),
                       "qpos": data.qpos.copy(), "anchor": xyzw(data.xquat[meta["robot"]["anchor_body_index"] + 1]),
                       "body": data.xpos[1:].copy(), "feet": data.xpos[body_ids].copy(),
                       "requestedFeet": targets[i].copy(), "stance": sample_stance[i].tolist()})
    hold = round(protocol["seconds"] / dt)
    for i in range(hold + 1, count):
        frames[i] = {key: value.copy() if isinstance(value, (np.ndarray, list)) else value
                     for key, value in frames[hold].items()}
        residuals[2 * i:2 * i + 2] = residuals[2 * hold:2 * hold + 2]
        for values in (torso_residuals, lowerings, penetrations, tendon_violations):
            values[i] = values[hold]
    qposes = np.asarray([frame["qpos"] for frame in frames])
    velocities = []
    for i in range(count):
        a, b = max(0, i - 1), min(count - 1, i + 1)
        velocity = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, velocity, (b - a) * dt, qposes[a], qposes[b])
        velocities.append(velocity)
        frames[i]["v"] = velocity[6:].astype(np.float32)
        frames[i]["qvel"] = velocity
    actual_feet = np.asarray([f["feet"] for f in frames])
    planted_pairs = sample_stance[:-1] & sample_stance[1:]
    stance_speeds = np.linalg.norm(np.diff(actual_feet[:, :, :2], axis=0) / dt, axis=2)[planted_pairs]
    clearances = [max((float(np.max(actual_feet[a:b, side, 2]) - settings["foot_origin_height_m"]))
                     for a, b in spans(~sample_stance[:, side])) for side in range(2)]
    residuals = np.asarray(residuals).reshape(count, 2, 2)
    gates = protocol["reference_gates"]
    metrics = {
        "source_frames": len(source), "source_duration_s": (len(source) - 1) / fps,
        "retarget_frames": count, "morphology_scale": scale, "source_metres_per_unit": skeleton.scale,
        "source_leg_lengths_m": source_lengths, "g1_leg_lengths_m": lengths,
        "ground_offset_m": ground_offset, "alignment_yaw_rad": float(heading),
        "maximum_root_reach_lowering_m": max(lowerings), "maximum_stance_adjustment_m": max(corrections),
        "source_contact_intervals_s": [[(a / fps, b / fps) for a, b in spans(stance[:, i])] for i in range(2)],
        "source_support_fraction": float(np.mean(stance.any(axis=1))),
        "max_foot_position_error_m": float(residuals[:, :, 0].max()),
        "max_foot_orientation_error_rad": float(residuals[:, :, 1].max()),
        "max_torso_orientation_error_rad": max(torso_residuals),
        "maximum_stance_slip_m_s": float(stance_speeds.max()) if len(stance_speeds) else None,
        "maximum_ground_penetration_m": max(penetrations),
        "max_joint_velocity_rad_s": float(np.max(np.abs(np.asarray(velocities)[:, 6:]))),
        "reference_swing_clearance_per_foot_m": clearances,
        "target_forward_progress_m": float(roots[round(protocol["seconds"] / dt), 0] - roots[0, 0]),
        "tendon_violation_frames": sum(tendon_violations),
    }
    checks = {
        "foot_position": metrics["max_foot_position_error_m"] <= gates["maximum_foot_position_error_m"],
        "foot_orientation": metrics["max_foot_orientation_error_rad"] <= gates["maximum_foot_orientation_error_rad"],
        "torso_orientation": metrics["max_torso_orientation_error_rad"] <= gates["maximum_torso_orientation_error_rad"],
        "joint_velocity": metrics["max_joint_velocity_rad_s"] <= gates["maximum_joint_velocity_rad_s"],
        "joint_limits": bool(np.all(qposes[:, 7:] >= model.jnt_range[1:, 0] - 1e-6) and np.all(qposes[:, 7:] <= model.jnt_range[1:, 1] + 1e-6)),
        "tendon_limits": not any(tendon_violations),
        "stationary_stance": metrics["maximum_stance_slip_m_s"] is not None and metrics["maximum_stance_slip_m_s"] <= gates["maximum_stance_slip_m_s"],
        "ground_clearance": metrics["maximum_ground_penetration_m"] <= gates["maximum_ground_penetration_m"],
        "bounded_contact_adjustment": max(corrections) <= gates["maximum_stance_adjustment_m"],
        "bounded_root_adjustment": max(lowerings) <= settings["maximum_root_reach_lowering_m"],
        "source_support": metrics["source_support_fraction"] >= gates["minimum_source_support_fraction"],
        "swing_clearance": all(value >= gates["minimum_reference_swing_clearance_m"] for value in clearances),
    }
    worst = np.argsort(np.max(residuals[:, :, 0], axis=1))[-5:][::-1]
    evidence = {
        "worstFootFitFrames": [{
            "time": int(i) * dt, "positionErrorPerFootM": residuals[i, :, 0].tolist(),
            "orientationErrorPerFootRad": residuals[i, :, 1].tolist(),
            "requestedFeetM": targets[i].tolist(), "realizedReferenceFeetM": actual_feet[i].tolist(),
            "stance": sample_stance[i].tolist(),
            "jointsAtLimits": [meta["robot"]["joint_names"][j] for j in range(29)
                              if min(qposes[i, j + 7] - model.jnt_range[j + 1, 0],
                                     model.jnt_range[j + 1, 1] - qposes[i, j + 7]) < 1e-4],
        } for i in worst],
        "sourceUnsupportedIntervalsS": [(a / fps, b / fps) for a, b in spans(~stance.any(axis=1))],
        "tendonViolationTimesS": [i * dt for i, value in enumerate(tendon_violations) if value],
    }
    return frames, {"passed": all(checks.values()), "gates": checks, "metrics": metrics, **evidence}, provenance


def run(assets: Path, cache: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    result = {"schemaVersion": 1, "kind": "cmu-mocap-reference-experiment", "identity": protocol["id"],
              "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol, "protocolSha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "sourceSha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("cmu.py", "retarget_cmu.py", "locomotion.py", "rollout.py")},
              "physicsExecuted": False}
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file)
    def save():
        temp = output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(output)
    try:
        result["modelProvenance"] = verify_assets(assets)
        directory = assets / MODEL_DIR
        meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        model = load_model(assets, meta, config)
        frames, diagnostics, result["motionProvenance"] = generate(model, meta, np.asarray(config["robot"]["default_dof_pos"]), protocol, cache)
        result["reference"] = diagnostics
        result["status"] = "reference_ready" if diagnostics["passed"] else "reference_failed"
        result["referenceEvidence"] = {
            "times": (np.arange(len(frames)) * meta["timing"]["control_dt"]).tolist(),
            "root": [f["root"].tolist() for f in frames], "feet": [f["feet"].tolist() for f in frames],
            "requestedFeet": [f["requestedFeet"].tolist() for f in frames],
            "stance": [f["stance"] for f in frames], "jointPositions": [f["q"].tolist() for f in frames],
            "jointVelocities": [f["v"].tolist() for f in frames],
        }
        save()
        print(json.dumps({"status": result["status"], "reference": diagnostics}, indent=2), flush=True)
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=Path("public/mocap/g1-cmu-07-04-v1.json"))
    args = parser.parse_args()
    result = run(args.assets, args.cache, args.output)
    if not result["reference"]["passed"]:
        raise SystemExit("Reference gates failed; no physics run is permitted. Evidence saved.")
