"""Bounded CPU-only G1 pretrained inference on self-authored reference poses.

The observation and PD protocol follows NVlabs/ProtoMotions' Apache-2.0
deployment/test_tracker_mujoco.py and deployment/tracker_inputs.py at the
revision in fetch_assets.py. No upstream executable code or pickle is loaded.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from fetch_assets import MODEL_DIR, MODEL_SHA256, REVISION

FALL_HEIGHT_M = 0.45
FALL_UPRIGHT_COS = 0.5
FALL_CONSECUTIVE_STEPS = 5


def verify_assets(root: Path) -> dict:
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if provenance["revision"] != REVISION:
        raise ValueError("Unexpected source revision.")
    for record in provenance["files"]:
        path = root / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Asset digest mismatch: {path}")
    onnx = root / MODEL_DIR / "compiled_models/unified_pipeline.onnx"
    if hashlib.sha256(onnx.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("The model does not match the pinned official LFS SHA-256.")
    return provenance


def load_model(assets: Path, meta: dict, config: dict) -> mujoco.MjModel:
    xml_path = assets / "protomotions/data/assets/mjcf/g1_holo_compat.xml"
    xml = ET.parse(xml_path).getroot()
    # Visual mesh geoms have zero density and no collisions. Inertias and all
    # primitive collision geoms remain intact, so STL downloads are unnecessary.
    for parent in xml.iter():
        for child in list(parent):
            if child.tag == "geom" and "mesh" in child.attrib:
                if child.get("contype") != "0" or child.get("conaffinity") != "0" or child.get("density") != "0":
                    raise ValueError("Refusing to remove a physical mesh geom.")
                parent.remove(child)
    asset = xml.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh":
                asset.remove(child)
    for sensor in xml.findall("sensor"):
        xml.remove(sensor)
    world = xml.find("worldbody")
    if world is None:
        raise ValueError("MJCF worldbody is missing.")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", rgba="0.2 0.3 0.35 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    joint_names = meta["robot"]["joint_names"]
    body_names = meta["robot"]["body_names"]
    if [model.joint(i).name for i in range(1, model.njnt)] != joint_names:
        raise ValueError("MuJoCo joint ordering differs from the policy contract.")
    if [model.body(i).name for i in range(1, model.nbody)] != body_names:
        raise ValueError("MuJoCo body ordering differs from the policy contract.")
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected a floating-base 29-DOF G1.")
    timing = meta["timing"]
    if not math.isclose(timing["control_dt"], timing["physics_dt"] * timing["decimation"]):
        raise ValueError("Control/physics timing contract mismatch.")
    if meta["control"]["action_ema_alpha"] != 1.0 or meta["control"]["pd_target_max_accel"] is not None:
        raise ValueError("This pinned runner supports only the shipped unfiltered PD contract.")
    model.opt.timestep = timing["physics_dt"]
    model.jnt_stiffness[:] = 0
    model.dof_damping[:] = 0
    model.dof_frictionloss[:] = 0
    for i, name in enumerate(joint_names):
        joint_id = model.joint(name).id
        if model.actuator(i).name != name or model.actuator_trnid[i, 0] != joint_id:
            raise ValueError(f"Actuator mapping mismatch: {name}")
        info = config["robot"]["control"]["control_info"][name]
        kp, kd = meta["control"]["stiffness"][i], meta["control"]["damping"][i]
        limit = float(info["effort_limit"])
        model.dof_armature[model.jnt_dofadr[joint_id]] = float(info["armature"])
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_biasprm[i, :3] = [0, -kp, -kd]
        model.actuator_ctrllimited[i] = False
        model.actuator_forcelimited[i] = True
        model.actuator_forcerange[i] = [-limit, limit]
    return model


def xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.asarray(wxyz)[..., [1, 2, 3, 0]].astype(np.float32)


def reference_pose(default: np.ndarray, names: list[str], t: float, case: str) -> tuple[np.ndarray, np.ndarray]:
    position = default.copy()
    velocity = np.zeros_like(position)
    if case == "arm_sweep":
        frequency = 2 * math.pi / 4.0
        for side in ("left", "right"):
            for suffix, amplitude in (("shoulder_pitch_joint", -0.15), ("elbow_joint", 0.25)):
                index = names.index(f"{side}_{suffix}")
                position[index] += amplitude * (1 - math.cos(frequency * t)) / 2
                velocity[index] = amplitude * frequency * math.sin(frequency * t) / 2
    elif case != "stand":
        raise ValueError(f"Unknown reference case: {case}")
    return position, velocity


def grounded_root_height(model: mujoco.MjModel, default: np.ndarray) -> float:
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, 0.8]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default
    mujoco.mj_forward(model, data)
    bottoms = []
    for i in range(model.ngeom):
        if "foot" not in (model.geom(i).name or "") or model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        vertical_extent = model.geom_size[i, 0] + model.geom_size[i, 1] * abs(data.geom_xmat[i, 8])
        bottoms.append(data.geom_xpos[i, 2] - vertical_extent)
    if not bottoms:
        raise ValueError("No primitive foot collision capsules found.")
    return float(0.8 - min(bottoms) + 0.002)


def make_inputs(data: mujoco.MjData, meta: dict, future: list[dict], previous_targets: np.ndarray) -> dict:
    anchor = meta["robot"]["anchor_body_index"] + 1
    values = {
        "current.anchor_rot": xyzw(data.xquat[anchor])[None],
        "current.dof_pos": data.qpos[7:].astype(np.float32)[None],
        "current.dof_vel": data.qvel[6:].astype(np.float32)[None],
        # MuJoCo's free-joint angular velocity is already pelvis-local.
        "current.root_local_ang_vel": data.qvel[3:6].astype(np.float32)[None],
        "historical.processed_actions": previous_targets.astype(np.float32)[None, None],
        "mimic.future_anchor_rot": np.stack([frame["anchor"] for frame in future])[None],
        "mimic.future_dof_pos": np.stack([frame["q"] for frame in future])[None],
        "mimic.future_dof_vel": np.stack([frame["v"] for frame in future])[None],
    }
    result = {}
    for name, key in meta["_runtime"]["onnx_name_to_in_key"].items():
        if key not in values:
            raise ValueError(f"Unsupported policy input: {key}")
        result[name] = np.ascontiguousarray(values[key], dtype=np.float32)
    return result


def make_session(path: Path, meta: dict) -> ort.InferenceSession:
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    declared = {item["name"]: item["shape"] for item in meta["policy_inputs"]}
    for tensor in session.get_inputs():
        shape = [1 if isinstance(n, str) or n is None else n for n in tensor.shape]
        if tensor.type != "tensor(float)" or declared.get(tensor.name) != shape:
            raise ValueError(f"Unexpected ONNX tensor contract: {tensor}")
    if set(declared) != {item.name for item in session.get_inputs()}:
        raise ValueError("ONNX and YAML inputs differ.")
    return session


@dataclass(frozen=True)
class Push:
    start_step: int
    end_step: int
    force_world_n: tuple[float, float, float]


def force_at_step(pushes: tuple[Push, ...], step: int) -> np.ndarray:
    force = np.zeros(3)
    for push in pushes:
        if push.start_step <= step < push.end_step:
            force += push.force_world_n
    return force


def controller_target(controller: str, session: ort.InferenceSession | None, data: mujoco.MjData,
                      meta: dict, reference: dict, future: list[dict], previous_targets: np.ndarray) -> tuple[np.ndarray, float | None]:
    if controller == "direct_pd":
        return reference["q"].copy(), None
    if controller != "onnx" or session is None:
        raise ValueError("Expected direct_pd or onnx with a loaded policy.")
    feed = make_inputs(data, meta, future, previous_targets)
    before = time.perf_counter()
    raw, target, kp, kd = session.run(["actions", "joint_pos_targets", "stiffness_targets", "damping_targets"], feed)
    inference_ms = (time.perf_counter() - before) * 1000
    if not all(np.isfinite(value).all() for value in (raw, target, kp, kd)):
        raise FloatingPointError("Non-finite policy output")
    if target.shape != (1, 29):
        raise ValueError("Unexpected policy output shape.")
    if not np.allclose(kp.reshape(-1), meta["control"]["stiffness"], rtol=1e-5) or not np.allclose(kd.reshape(-1), meta["control"]["damping"], rtol=1e-5):
        raise ValueError("Baked ONNX gains disagree with configured physics gains.")
    return target[0].copy(), inference_ms


def run_case(model: mujoco.MjModel, session: ort.InferenceSession | None, meta: dict, default: np.ndarray,
             case: str, seconds: float, *, controller: str = "onnx", pushes: tuple[Push, ...] = (),
             record_frames: bool = True, wall_limit_seconds: float | None = None,
             reference_frames: list[dict] | None = None,
             observe: Callable[[mujoco.MjModel, mujoco.MjData, int, dict], None] | None = None,
             observe_substep: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
             initial_state: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    if controller not in ("onnx", "direct_pd") or (controller == "onnx" and session is None):
        raise ValueError("Invalid controller or missing policy.")
    if not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError("Each rollout must be between 1 and 30 seconds.")
    if wall_limit_seconds is not None and (not math.isfinite(wall_limit_seconds) or wall_limit_seconds <= 0):
        raise ValueError("Wall-time budget must be positive and finite.")
    dt = meta["timing"]["control_dt"]
    steps = round(seconds / dt)
    for push in pushes:
        if not 0 <= push.start_step < push.end_step <= steps or len(push.force_world_n) != 3 or not np.isfinite(push.force_world_n).all():
            raise ValueError("Invalid disturbance schedule.")
    height = grounded_root_height(model, default)
    anchor = meta["robot"]["anchor_body_index"] + 1
    reference_data = mujoco.MjData(model)
    references = []
    required_frames = steps + max(meta["motion"]["future_step_indices"]) + 1
    for i in range(required_frames if reference_frames is None else 0):
        q, v = reference_pose(default, meta["robot"]["joint_names"], i * dt, case)
        reference_data.qpos[:3] = [0, 0, height]
        reference_data.qpos[3:7] = [1, 0, 0, 0]
        reference_data.qpos[7:] = q
        mujoco.mj_kinematics(model, reference_data)
        references.append({
            "q": q.astype(np.float32), "v": v.astype(np.float32),
            "anchor": xyzw(reference_data.xquat[anchor]),
            "body": reference_data.xpos[1:].copy(),
        })
    if reference_frames is not None:
        if len(reference_frames) < required_frames:
            raise ValueError("Reference must cover the rollout and policy lookahead.")
        for frame in reference_frames:
            for key, shape in (("q", (29,)), ("v", (29,)), ("anchor", (4,)), ("body", (33, 3))):
                value = np.asarray(frame[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid reference {key}.")
        if initial_state is None and (not np.allclose(reference_frames[0]["q"], default, atol=1e-6)
                                      or not np.allclose(reference_frames[0]["body"][0], [0, 0, height], atol=1e-6)):
            raise ValueError("Reference must start at the unchanged nominal initial state.")
        references = reference_frames
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, height]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default
    if initial_state is not None:
        if reference_frames is None:
            raise ValueError("Explicit initial state requires matching custom reference frames.")
        position, velocity = (np.asarray(value) for value in initial_state)
        if (position.shape != (model.nq,) or velocity.shape != (model.nv,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or abs(np.linalg.norm(position[3:7]) - 1) > 1e-6 or position[2] < FALL_HEIGHT_M
            or np.any(position[7:] < model.jnt_range[1:, 0]) or np.any(position[7:] > model.jnt_range[1:, 1])):
            raise ValueError("Invalid explicit initial pose/velocity.")
        data.qpos[:], data.qvel[:] = position, velocity
    mujoco.mj_forward(model, data)
    if initial_state is not None:
        first = references[0]
        if (not np.allclose(data.qpos[7:], first["q"], atol=1e-6)
            or not np.allclose(data.qvel[6:], first["v"], atol=1e-6)
            or not np.allclose(data.xpos[1:], first["body"], atol=1e-6)
            or not np.allclose(xyzw(data.xquat[anchor]), first["anchor"], atol=1e-6)
            or np.any(data.ten_length < model.tendon_range[:, 0] - 1e-6)
            or np.any(data.ten_length > model.tendon_range[:, 1] + 1e-6)):
            raise ValueError("Explicit initial state disagrees with reference or ankle constraints.")
        height = float(data.qpos[2])
    previous_targets = np.zeros(29, dtype=np.float32)
    frames, errors, body_errors, inference_ms, control_ms = [], [], [], [], []
    heights, tilts, contacts, target_changes = [], [], [], []
    first_fall_time = None
    fall_streak = 0
    warnings_before = data.warning.number.copy()
    completed_steps = physics_steps = controller_calls = 0
    termination = None
    started = time.perf_counter()
    for step in range(steps + 1):
        reference = references[step]
        if observe is not None:
            observe(model, data, step, reference)
        joint_error = float(np.sqrt(np.mean((data.qpos[7:] - reference["q"]) ** 2)))
        body_error = float(np.mean(np.linalg.norm(data.xpos[1:] - reference["body"], axis=1)))
        pelvis_up = float(data.xmat[1, 8])
        root_height = float(data.qpos[2])
        fall_streak = fall_streak + 1 if root_height < FALL_HEIGHT_M or pelvis_up < FALL_UPRIGHT_COS else 0
        if first_fall_time is None and fall_streak >= FALL_CONSECUTIVE_STEPS:
            first_fall_time = (step - FALL_CONSECUTIVE_STEPS + 1) * dt
        errors.append(joint_error)
        body_errors.append(body_error)
        heights.append(root_height)
        tilts.append(math.degrees(math.acos(float(np.clip(pelvis_up, -1, 1)))))
        contacts.append(int(data.ncon))
        if record_frames and (step % 2 == 0 or step == steps):
            frames.append({
                "time": round(step * dt, 6),
                "body": np.round(data.xpos[1:], 5).tolist(),
                "reference": np.round(reference["body"], 5).tolist(),
                "jointRmsRad": round(joint_error, 6),
                "rootHeightM": round(root_height, 6),
                "contacts": int(data.ncon),
                "fallen": first_fall_time is not None,
            })
        if step == steps:
            break
        if wall_limit_seconds is not None and time.perf_counter() - started >= wall_limit_seconds:
            termination = "wall_budget_exceeded"
            break
        future = [references[step + offset] for offset in meta["motion"]["future_step_indices"]]
        before = time.perf_counter()
        controller_calls += 1
        try:
            target, latency = controller_target(controller, session, data, meta, reference, future, previous_targets)
        except FloatingPointError as error:
            termination = str(error)
            break
        control_ms.append((time.perf_counter() - before) * 1000)
        if latency is not None:
            inference_ms.append(latency)
        target_changes.append(float(np.linalg.norm(target - previous_targets)))
        previous_targets = target
        data.ctrl[:] = previous_targets
        data.xfrc_applied[:] = 0
        data.xfrc_applied[1, :3] = force_at_step(pushes, step)
        for _ in range(meta["timing"]["decimation"]):
            mujoco.mj_step(model, data)
            physics_steps += 1
            if observe_substep is not None:
                observe_substep(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            termination = f"Non-finite physics after control step {step}"
            break
        if np.any(data.warning.number > warnings_before):
            termination = f"MuJoCo numerical warning after control step {step}"
            break
        completed_steps += 1
    wall_seconds = time.perf_counter() - started
    metrics = {
        "requested_seconds": seconds,
        "control_steps": completed_steps,
        "controller_calls": controller_calls,
        "physics_steps": physics_steps,
        "error_samples": len(errors),
        "simulated_seconds": round(completed_steps * dt, 6),
        "completed": termination is None and completed_steps == steps and math.isclose(float(data.time), steps * dt, abs_tol=1e-7),
        "fell": first_fall_time is not None,
        "first_fall_time_s": first_fall_time,
        "joint_rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
        "max_frame_joint_rms_rad": max(errors),
        "mean_body_position_error_m": float(np.mean(body_errors)),
        "min_root_height_m": min(heights),
        "max_pelvis_tilt_deg": max(tilts),
        "fraction_frames_with_contacts": float(np.mean(np.asarray(contacts) > 0)),
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
        "p95_inference_ms": float(np.percentile(inference_ms, 95)) if inference_ms else None,
        "mean_control_ms": float(np.mean(control_ms)) if control_ms else None,
        "mean_pd_target_change_l2_rad": float(np.mean(target_changes[1:])) if len(target_changes) > 1 else None,
        "wall_seconds": wall_seconds,
        "simulation_to_wall_ratio": completed_steps * dt / wall_seconds,
        "numerical_warnings": int(np.sum(data.warning.number - warnings_before)),
        "initial_root_height_m": height,
    }
    print(json.dumps({"case": case, "controller": controller, "termination": termination, **metrics}), flush=True)
    return {"name": case, "controller": controller, "termination": termination, "metrics": metrics, "frames": frames}


def run(assets: Path, output: Path, seconds: float) -> dict:
    if not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError("Each rollout must be between 1 and 30 seconds.")
    provenance = verify_assets(assets)
    model_dir = assets / MODEL_DIR
    meta = yaml.safe_load((model_dir / "compiled_models/unified_pipeline.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((model_dir / "resolved_configs_inference.yaml").read_text(encoding="utf-8"))
    default = np.asarray(config["robot"]["default_dof_pos"], dtype=np.float64)
    model = load_model(assets, meta, config)
    session = make_session(model_dir / "compiled_models/unified_pipeline.onnx", meta)
    result = {
        "schemaVersion": 1,
        "kind": "recorded-pretrained-policy-physics",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "policy": "NVIDIA G1 BONES-SEED deployment tracker",
        "reference": "Self-authored grounded nominal stand and symmetric slow arm sweep; no motion dataset used.",
        "provenance": provenance,
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "onnxruntime", "PyYAML")},
            "provider": "CPUExecutionProvider", "inference_threads": 1,
            "physics_dt": meta["timing"]["physics_dt"], "control_dt": meta["timing"]["control_dt"],
            "integrator": "MuJoCo default Euler", "visual_meshes_loaded": False,
            "joint_torque_limits_enforced": True,
        },
        "fallDefinition": {
            "root_height_below_m": FALL_HEIGHT_M,
            "pelvis_upright_cos_below": FALL_UPRIGHT_COS,
            "consecutive_control_samples": FALL_CONSECUTIVE_STEPS,
        },
        "limitations": [
            "Two deterministic synthetic-reference smoke tests, not a dataset benchmark or SOTA result.",
            "No training, real robot, domain randomization or generalization measurement.",
            "Browser shows recorded MuJoCo body transforms, not live browser policy inference.",
            "Collision count includes all MuJoCo contacts, not just support-foot contacts.",
            "Visual-only zero-density meshes omitted; inertias and primitive collision geometry preserved.",
        ],
        "bodyNames": meta["robot"]["body_names"],
        "parents": (model.body_parentid[1:] - 1).tolist(),
        "cases": [run_case(model, session, meta, default, case, seconds) for case in ("stand", "arm_sweep")],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    if any(not case["metrics"]["completed"] for case in result["cases"]):
        raise RuntimeError(f"Incomplete rollout saved to {output}; inspect termination and metrics.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--output", type=Path, default=Path("public/physics-replay.json"))
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    run(args.assets, args.output, args.seconds)
