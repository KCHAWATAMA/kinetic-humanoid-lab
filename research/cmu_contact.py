"""Contact-aware CMU reference repair, with immutable versioned outcomes."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from cmu import CACHE, forward, load
from constrained_ik import ConstrainedIK, PoseTargets
from fetch_assets import MODEL_DIR
from locomotion import FEET, LocomotionObserver, smooth
from retarget_cmu import contacts, filtered, quaternion, slerp, spans
from rollout import load_model, make_session, run_case, verify_assets, xyzw

PROTOCOL = Path(__file__).with_name("cmu-contact-protocol.json")


def infer_contacts(sole, fps, settings):
    velocity = np.gradient(sole, 1 / fps, axis=0)
    mask = ((np.linalg.norm(velocity[:, :2], axis=1) < settings["contact_horizontal_speed_m_s"])
            & (np.abs(velocity[:, 2]) < settings["contact_vertical_speed_m_s"]))
    return contacts(mask, fps, settings["maximum_contact_gap_s"], settings["minimum_source_stance_s"])


def blend_swing_corrections(raw, corrected, stance):
    intervals = spans(stance)
    if not intervals:
        raise ValueError("Source foot has no inferred stance.")
    gaps = [(0, intervals[0][0], None, intervals[0][0])]
    gaps += [(intervals[i][1], intervals[i + 1][0], intervals[i][1] - 1, intervals[i + 1][0])
             for i in range(len(intervals) - 1)]
    gaps += [(intervals[-1][1], len(raw), intervals[-1][1] - 1, None)]
    for start, end, a, b in gaps:
        a = b if a is None else a
        b = a if b is None else b
        da, db = corrected[a] - raw[a], corrected[b] - raw[b]
        for i in range(start, end):
            weight = smooth((i - a) / (b - a)) if b > a else 0.
            corrected[i] = raw[i] + (1 - weight) * da + weight * db
    return corrected


def grounded_rocking_feet(raw, rotations, stance, points, radii, padding):
    result = raw.copy()
    for a, b in spans(stance):
        relative = np.zeros((b - a, 2))
        previous = rotations[a]
        pivot = points[np.argmin((points @ previous.T)[:, 2] - radii)]
        for i in range(a, b):
            current = rotations[i]
            if i > a:
                relative[i - a] = relative[i - a - 1] + ((previous - current) @ pivot)[:2]
            pivot = points[np.argmin((points @ current.T)[:, 2] - radii)]
            previous = current
        origin = np.median(raw[a:b, :2] - relative, axis=0)
        result[a:b, :2] = origin + relative
        result[a:b, 2] = [padding - np.min((points @ r.T)[:, 2] - radii) for r in rotations[a:b]]
    result = blend_swing_corrections(raw, result, stance)
    for i, rotation in enumerate(rotations):
        result[i, 2] = max(result[i, 2], padding - float(np.min((points @ rotation.T)[:, 2] - radii)))
    return result


def source_targets(model, default, protocol, cache=CACHE):
    skeleton, samples, provenance = load(cache)
    poses = [forward(skeleton, frame) for frame in samples]
    fps, settings = protocol["source_fps"], protocol["retarget"]
    names = ("root", "lhipjoint", "rhipjoint", "ltibia", "rtibia", "lfoot", "rfoot", "ltoes", "rtoes")
    p = {name: filtered(np.asarray([position[name] for position, _ in poses])) for name in names}
    basis = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
    travel = basis @ (p["root"][-1] - p["root"][0])
    heading = np.arctan2(travel[1], travel[0])
    world = Rotation.from_euler("z", -heading).as_matrix() @ basis
    p = {name: values @ world.T for name, values in p.items()}
    origin = p["root"][0] * [1, 1, 0]
    p = {name: values - origin for name, values in p.items()}
    rotations = {name: np.asarray([world @ r[name] @ basis.T for _, r in poses]) for name in ("root", "thorax", "lfoot", "rfoot")}
    stance = np.stack([infer_contacts((p[side + "foot"] + p[side + "toes"]) / 2, fps, settings)
                       for side in ("l", "r")], axis=1)
    solver = ConstrainedIK(model, default)
    data = mujoco.MjData(model)
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = 0, [1, 0, 0, 0], default
    mujoco.mj_forward(model, data)
    hips, lengths, human_lengths = [], [], []
    for side, prefix in (("left", "l"), ("right", "r")):
        hip, knee, ankle = [data.xpos[model.body(side + suffix).id].copy()
                            for suffix in ("_hip_roll_link", "_knee_link", "_ankle_roll_link")]
        hips.append(hip)
        lengths.append(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle))
        human_lengths.append(sum(skeleton.bones[prefix + part].length * skeleton.scale for part in ("femur", "tibia")))
    scale = float(np.mean(np.asarray(lengths) / human_lengths))
    roots = p["root"] * scale
    raw_feet = np.stack([roots + rotations["root"] @ hips[i] + scale * (p[side + "tibia"] - p[side + "hipjoint"])
                         for i, side in enumerate(("l", "r"))], axis=1)
    feet_rotations, calibration = [], []
    for i, side in enumerate(("l", "r")):
        angles = Rotation.from_matrix(rotations[side + "foot"]).as_euler("ZYX")
        angles[:, 0] = np.unwrap(angles[:, 0])
        angles = filtered(angles)
        middle = np.zeros(len(samples), dtype=bool)
        yaw = angles[:, 0].copy()
        for a, b in spans(stance[:, i]):
            margin = (b - a) // 4
            middle[a + margin:b - margin] = True
            yaw[a:b] = np.median(angles[a:b, 0])
        if not middle.any():
            raise ValueError("Foot pitch calibration needs a mid-stance interval.")
        neutral_pitch = float(np.median(angles[middle, 1]))
        yaw = blend_swing_corrections(angles[:, 0], yaw, stance[:, i])
        pitch = angles[:, 1] - neutral_pitch
        feet_rotations.append(Rotation.from_euler("ZY", np.c_[yaw, pitch]).as_matrix())
        calibration.append(neutral_pitch)
    feet_rotations = np.stack(feet_rotations, axis=1)
    required_height = np.stack([[settings["floor_padding_m"] - np.min((points @ r.T)[:, 2] - radii)
                                 for r in feet_rotations[:, i]] for i, (points, radii) in enumerate(solver.capsules)], axis=1)
    ground_offset = float(np.median((required_height - raw_feet[:, :, 2])[stance]))
    roots[:, 2] += ground_offset
    raw_feet[:, :, 2] += ground_offset
    feet = np.stack([grounded_rocking_feet(raw_feet[:, i], feet_rotations[:, i], stance[:, i], points, radii,
                                         settings["floor_padding_m"])
                     for i, (points, radii) in enumerate(solver.capsules)], axis=1)
    dt = protocol["control_dt"]
    times = np.arange(round(protocol["seconds"] / dt) + 1) * dt
    source_time = np.arange(len(samples)) / fps
    def interpolate(values):
        shape = values.shape[1:]
        flat = values.reshape(len(values), -1)
        return np.stack([np.interp(times, source_time, flat[:, j]) for j in range(flat.shape[1])], axis=1).reshape((len(times), *shape))
    def interpolate_rotations(values):
        quaternions = [quaternion(r) for r in values]
        return np.asarray([slerp(quaternions[int(t * fps)], quaternions[min(int(t * fps) + 1, len(samples) - 1)],
                                 t * fps % 1) for t in times])
    roots, feet = interpolate(roots), interpolate(feet)
    root_q = interpolate_rotations(rotations["root"])
    torso_q = interpolate_rotations(rotations["thorax"])
    foot_q = [interpolate_rotations(feet_rotations[:, i]) for i in range(2)]
    def matrix(q):
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    targets = [PoseTargets(roots[i], root_q[i], feet[i], np.stack([matrix(q[i]) for q in foot_q]), matrix(torso_q[i]))
               for i in range(len(times))]
    contact = np.asarray([stance[int(t * fps)] & stance[min(int(np.ceil(t * fps)), len(samples) - 1)] for t in times])
    summary = {"source_frames": len(samples), "morphology_scale": scale, "ground_offset_m": ground_offset,
               "neutral_foot_pitch_rad": calibration, "source_support_fraction": float(stance.any(axis=1).mean()),
               "source_contact_intervals_s": [[(a / fps, b / fps) for a, b in spans(stance[:, i])] for i in range(2)],
               "maximum_source_ankle_adjustment_m": float(np.linalg.norm(feet - interpolate(raw_feet), axis=2).max())}
    return targets, contact, summary, provenance


def fit_valid(report, gates):
    return (report["optimizer_success"] and max(report["foot_error_m"]) <= gates["maximum_foot_position_error_m"]
            and max(report["foot_angle_error_rad"]) <= gates["maximum_foot_orientation_error_rad"]
            and report["torso_angle_error_rad"] <= gates["maximum_torso_orientation_error_rad"]
            and report["floor_min_m"] >= -gates["maximum_ground_penetration_m"]
            and report["tendon_max_violation"] <= 1e-6)


def generate(model, meta, default, protocol, progress, cache=CACHE):
    targets, stance, source, provenance = source_targets(model, default, protocol, cache)
    solver = ConstrainedIK(model, default)
    frames, reports, seed = [], [], None
    start = time.perf_counter()
    gates, settings = protocol["reference_gates"], protocol["retarget"]
    for i, target in enumerate(targets):
        if time.perf_counter() - start > protocol["reference_wall_limit_seconds"]:
            return frames, {"passed": False, "error": "Reference generation wall budget exceeded.", "source": source,
                            "solver": reports}, provenance
        attempts = []
        for guess in ([seed, None] if seed is not None else [None]):
            x, qpos, report = solver.solve(target, guess, root_bound=np.asarray(settings["root_translation_bound_m"]),
                                           max_iterations=settings["solver_iterations"],
                                           seed_evaluations=settings["bounded_least_squares_seed_evaluations"],
                                           previous=seed, max_joint_step=gates["maximum_joint_velocity_rad_s"] * protocol["control_dt"] * (1 - 1e-7))
            attempts.append(report)
            if fit_valid(report, gates):
                break
        reports.append({"time": i * protocol["control_dt"], "starts": attempts})
        if not fit_valid(report, gates):
            return frames, {"passed": False, "error": f"No valid constrained fit at frame {i}; bad warm start not propagated.",
                            "source": source, "solver": reports}, provenance
        seed = x
        data = solver.data
        frames.append({"qpos": qpos, "q": qpos[7:].astype(np.float32), "root": qpos[:3].copy(),
                       "anchor": xyzw(data.xquat[meta["robot"]["anchor_body_index"] + 1]),
                       "body": data.xpos[1:].copy(), "stance": stance[i].tolist()})
        if i % 25 == 0:
            progress(i, reports[-1])
    hold = len(frames) - 1
    for _ in range(max(meta["motion"]["future_step_indices"])):
        frames.append({key: value.copy() for key, value in frames[hold].items()})
    dt = protocol["control_dt"]
    data, jacp = mujoco.MjData(model), np.zeros((3, model.nv))
    slip, stance_gaps, clearances = [[], []], [], [[], []]
    velocities = []
    for i, frame in enumerate(frames):
        a, b = max(0, i - 1), min(len(frames) - 1, i + 1)
        velocity = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, velocity, (b - a) * dt, frames[a]["qpos"], frames[b]["qpos"])
        frame["qvel"], frame["v"] = velocity, velocity[6:].astype(np.float32)
        velocities.append(velocity)
        data.qpos[:], data.qvel[:] = frame["qpos"], velocity
        mujoco.mj_forward(model, data)
        for side, (points, radii) in enumerate(solver.capsules):
            body = solver.bodies[side]
            world = data.xpos[body] + points @ data.xmat[body].reshape(3, 3).T
            bottom = world[:, 2] - radii
            if i <= hold and frame["stance"][side]:
                stance_gaps.append(float(bottom.min()))
                if 0 < i < hold and frames[i - 1]["stance"][side] and frames[i + 1]["stance"][side]:
                    active = np.flatnonzero(bottom <= bottom.min() + .001)
                    speeds = []
                    for point_index in active:
                        contact = world[point_index] - [0, 0, radii[point_index]]
                        mujoco.mj_jac(model, data, jacp, None, contact, body)
                        speeds.append(np.linalg.norm((jacp @ velocity)[:2]))
                    slip[side].append(float(max(speeds)))
            elif i <= hold:
                clearances[side].append(float(bottom.min()))
    velocity = np.asarray(velocities)
    metrics = {
        **source, "frames": len(frames), "max_joint_velocity_rad_s": float(np.abs(velocity[:, 6:]).max()),
        "max_root_speed_m_s": float(np.linalg.norm(velocity[:, :3], axis=1).max()),
        "maximum_stance_gap_m": max(stance_gaps) if stance_gaps else None,
        "mean_stance_point_speed_m_s": [float(np.mean(v)) if v else None for v in slip],
        "p95_stance_point_speed_m_s": [float(np.percentile(v, 95)) if v else None for v in slip],
        "swing_clearance_m": [max(v) if v else 0. for v in clearances],
        "target_forward_progress_m": float(frames[hold]["root"][0] - frames[0]["root"][0]),
        "maximum_foot_fit_error_m": max(max(r["starts"][-1]["foot_error_m"]) for r in reports),
        "maximum_foot_fit_angle_rad": max(max(r["starts"][-1]["foot_angle_error_rad"]) for r in reports),
        "generation_wall_seconds": time.perf_counter() - start,
    }
    checks = {
        "constrained_fits": True, "source_support": source["source_support_fraction"] >= gates["minimum_source_support_fraction"],
        "joint_speed": metrics["max_joint_velocity_rad_s"] <= gates["maximum_joint_velocity_rad_s"],
        "root_speed": metrics["max_root_speed_m_s"] <= gates["maximum_root_speed_m_s"],
        "grounded_stance": metrics["maximum_stance_gap_m"] is not None and metrics["maximum_stance_gap_m"] <= gates["maximum_stance_gap_m"],
        "mean_stance_motion": all(v is not None and v <= gates["maximum_mean_stance_point_speed_m_s"] for v in metrics["mean_stance_point_speed_m_s"]),
        "p95_stance_motion": all(v is not None and v <= gates["maximum_p95_stance_point_speed_m_s"] for v in metrics["p95_stance_point_speed_m_s"]),
        "swing_clearance": all(v >= gates["minimum_reference_swing_clearance_m"] for v in metrics["swing_clearance_m"]),
    }
    return frames, {"passed": all(checks.values()), "gates": checks, "metrics": metrics, "solver": reports}, provenance


def run(assets, output, execute_physics):
    protocol = json.loads(PROTOCOL.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schemaVersion": 1, "kind": "cmu-contact-retarget-experiment", "identity": protocol["id"],
              "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "protocolSha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "physicsExecuted": False,
              "sourceSha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("cmu.py", "cmu_contact.py", "constrained_ik.py", "retarget_cmu.py", "rollout.py", "locomotion.py")},
              "runtime": {"python": platform.python_version(),
                          "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mujoco", "onnxruntime")}}}
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
        model, default = load_model(assets, meta, config), np.asarray(config["robot"]["default_dof_pos"])
        def progress(frame, report):
            result["progress"] = {"frame": frame, "lastFit": report}
            save()
        frames, reference, result["motionProvenance"] = generate(model, meta, default, protocol, progress)
        result["reference"] = reference
        result["referenceEvidence"] = {key: [np.asarray(frame[key]).tolist() for frame in frames]
                                       for key in ("qpos", "root", "body", "stance")}
        if reference["passed"]:
            result["referenceEvidence"]["qvel"] = [frame["qvel"].tolist() for frame in frames]
        result["status"] = "reference_ready" if reference["passed"] else "reference_failed"
        save()
        if reference["passed"] and execute_physics:
            physical_protocol = json.loads(json.dumps(protocol))
            progress_m = reference["metrics"]["target_forward_progress_m"]
            physical_protocol["criteria"].update(minimum_forward_progress_m=progress_m * protocol["progress_gate_ratio"][0],
                                                  maximum_forward_progress_m=progress_m * protocol["progress_gate_ratio"][1])
            result["physicalProtocol"] = physical_protocol
            result["initialState"] = {"qpos": frames[0]["qpos"].tolist(), "qvel": frames[0]["qvel"].tolist()}
            result["status"], result["physicsExecuted"] = "physics_running", True
            save()
            session = make_session(directory / "compiled_models/unified_pipeline.onnx", meta)
            observer = LocomotionObserver(physical_protocol, meta["timing"]["control_dt"])
            physics = run_case(model, session, meta, default, protocol["id"], protocol["seconds"],
                               reference_frames=frames, observe=observer, wall_limit_seconds=protocol["wall_limit_seconds"],
                               initial_state=(frames[0]["qpos"], frames[0]["qvel"]))
            result["physics"] = physics
            result["locomotion"] = observer.summarize(physics["metrics"])
            result["bodyNames"], result["parents"] = meta["robot"]["body_names"], (model.body_parentid[1:] - 1).tolist()
            result["status"] = "complete"
            save()
        print(json.dumps({"status": result["status"], "reference": {k: v for k, v in reference.items() if k != "solver"},
                          "locomotion": {k: v for k, v in result.get("locomotion", {}).items() if k != "samples"}}, indent=2), flush=True)
        return result
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--output", type=Path, default=Path("public/mocap-v2/attempt-03.json"))
    parser.add_argument("--physics", action="store_true")
    args = parser.parse_args()
    result = run(args.assets, args.output, args.physics)
    if not result["reference"]["passed"] or (args.physics and not result.get("locomotion", {}).get("passed")):
        raise SystemExit("Fixed gates failed; complete evidence retained.")
