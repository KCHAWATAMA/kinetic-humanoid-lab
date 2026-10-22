"""Read-only contract instrumentation and bounded, unchanged-reference diagnosis.

Observation expectations are independently expressed with rotation matrices from
the pinned humanoid.py, target_poses.py and rotations.py semantics. Upstream
Python is fetched as text for provenance, never imported or executed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import yaml

from fetch_assets import MODEL_DIR, RAW, REVISION, download
from locomotion import FEET, PROTOCOL_PATH, contact_telemetry, generate_reference
from rollout import load_model, make_inputs, make_session, run_case, verify_assets, xyzw

PROTOCOL = Path(__file__).with_name("diagnosis-protocol.json")
SOURCE_PATHS = (
    "deployment/test_tracker_mujoco.py",
    "deployment/tracker_inputs.py",
    "deployment/motion_utils.py",
    "protomotions/envs/obs/humanoid.py",
    "protomotions/envs/obs/target_poses.py",
    "protomotions/utils/rotations.py",
    "protomotions/simulator/mujoco/simulator.py",
    "docs/source/tutorials/workflows/g1_deployment.rst",
    "docs/source/getting_started/seed_g1_csv_preparation.rst",
)
HISTORICAL = (
    "public/physics-replay.json", "public/physics-comparison.json",
    "public/locomotion/attempt-01.json", "public/locomotion/attempt-02.json",
    "public/locomotion/summary.json",
)
RAW_OBS = "/policy_module/mu/Concat_output_0"
NORMALIZED_OBS = "/policy_module/mu/norm/Div_output_0"
CLIPPED_OBS = "/policy_module/mu/norm/Clip_output_0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_records(cache: Path, fetch: bool) -> list[dict]:
    records = []
    for relative in SOURCE_PATHS:
        path = cache / (relative + ".txt")
        if fetch:
            payload = download(RAW + relative, 200_000)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if not path.exists():
            raise FileNotFoundError(f"Missing source text {path}; run with --fetch-sources.")
        records.append({"url": RAW + relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    return records


def rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_xyzw, dtype=np.float64)[[3, 0, 1, 2]])
    return matrix.reshape(3, 3)


def expected_observation(feed: dict) -> np.ndarray:
    current_rotation = rotation_matrix(feed["current_anchor_rot"][0])
    proprio = np.concatenate((
        feed["current_dof_pos"][0], feed["current_dof_vel"][0],
        feed["current_root_local_ang_vel"][0], current_rotation.T @ [0, 0, -1],
    ))
    targets = []
    for index, quaternion in enumerate(feed["mimic_future_anchor_rot"][0]):
        relative = current_rotation.T @ rotation_matrix(quaternion)
        targets.extend((
            relative[:, 0], relative[:, 2],
            feed["mimic_future_dof_vel"][0, index],
            feed["mimic_future_dof_pos"][0, index],
        ))
    return np.concatenate((proprio, *targets, feed["historical_processed_actions"][0, 0])).astype(np.float32)[None]


def inspect_graph(path: Path, meta: dict, config: dict):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph)
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    ancestry = {item.name: {item.name} for item in graph.graph.input}
    for node in graph.graph.node:
        dependencies = set().union(*(ancestry.get(name, set()) for name in node.input))
        for name in node.output:
            ancestry[name] = dependencies
    action_nodes = [node for node in graph.graph.node if node.name in ("/action_module/Mul", "/action_module/Add")]
    if [(node.op_type, list(node.input)) for node in action_nodes] != [
        ("Mul", ["action_module.action_scale", "actions"]),
        ("Add", ["action_module.pd_action_offset", "/action_module/Mul_output_0"]),
    ]:
        raise ValueError("Pinned action-transform graph changed.")
    offset, scale = (initializers["action_module." + name] for name in ("pd_action_offset", "action_scale"))
    action_config = config["env"]["action_config"]
    np.testing.assert_allclose(offset, action_config["pd_action_offset"], atol=1e-7)
    np.testing.assert_allclose(scale, action_config["action_scale"], atol=1e-7)
    np.testing.assert_allclose(offset, config["robot"]["default_dof_pos"], atol=1e-7)
    for tensor in graph.graph.input:
        if tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Expected float32 graph input.")
    for name in (RAW_OBS, NORMALIZED_OBS, CLIPPED_OBS):
        graph.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 349]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort.disable_telemetry_events()
    instrumented = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
    report = {
        "inputMetadata": meta["policy_inputs"],
        "actionStructuralAncestors": sorted(ancestry["actions"]),
        "passthroughNotGraphInputs": meta["_runtime"]["passthrough_keys"],
        "observationFeatures": 349,
        "normalization": "(observation - baked_mean) / baked_denominator, then clip [-5,5]",
        "actionTransform": "offset + scale * actions; no Tanh or action Clip in this pinned graph",
        "tanhNodeCount": sum(node.op_type == "Tanh" for node in graph.graph.node),
        "pdOffset": offset.tolist(), "actionScale": scale.tolist(),
        "warning": "Structural ancestry includes shape-only paths; numeric probes establish value sensitivity. Gain outputs are constants broadcast using batch shape.",
    }
    report["inputs"] = [{"name": item.name, "dtype": item.type, "shape": item.shape} for item in instrumented.get_inputs()]
    return instrumented, initializers, report


def probe(instrumented, initializers: dict, ordinary, feed: dict) -> dict:
    raw, normalized, clipped = instrumented.run([RAW_OBS, NORMALIZED_OBS, CLIPPED_OBS], feed)
    expected = expected_observation(feed)
    np.testing.assert_allclose(raw, expected, atol=2e-6, rtol=1e-6)
    expected_norm = (raw - initializers["onnx::Sub_787"]) / initializers["onnx::Div_790"]
    np.testing.assert_allclose(normalized, expected_norm, atol=2e-5, rtol=2e-6)
    np.testing.assert_allclose(clipped, np.clip(expected_norm, -5, 5), atol=2e-5, rtol=2e-6)
    names = ["actions", "joint_pos_targets", "stiffness_targets", "damping_targets"]
    outputs = ordinary.run(names, feed)
    instrumented_outputs = instrumented.run(names, feed)
    for actual, check in zip(outputs, instrumented_outputs):
        np.testing.assert_allclose(actual, check, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(outputs[1], initializers["action_module.pd_action_offset"]
                               + initializers["action_module.action_scale"] * outputs[0], atol=2e-6)
    return {
        "observationMaxError": float(np.max(np.abs(raw - expected))),
        "clippedFeatureCount": int(np.count_nonzero(np.abs(normalized) > 5)),
        "maximumAbsNormalizedFeature": float(np.max(np.abs(normalized))),
        "rawObservation": raw[0].tolist(), "normalizedObservation": normalized[0].tolist(),
        "target": outputs[1][0].tolist(),
    }


class CommandTrace:
    def __init__(self, session, references: list[dict] | None, meta: dict):
        self.session, self.references, self.meta = session, references, meta
        self.targets: list[np.ndarray] = []
        self.max_history_error = self.max_future_error = 0.0

    def run(self, names, feed):
        expected_previous = self.targets[-1] if self.targets else np.zeros(29)
        self.max_history_error = max(self.max_history_error, float(np.max(np.abs(
            feed["historical_processed_actions"][0, 0] - expected_previous))))
        if self.references is not None:
            for slot, offset in enumerate(self.meta["motion"]["future_step_indices"]):
                frame = self.references[len(self.targets) + offset]
                for name, key in (("mimic_future_dof_pos", "q"), ("mimic_future_dof_vel", "v"),
                                  ("mimic_future_anchor_rot", "anchor")):
                    self.max_future_error = max(self.max_future_error, float(np.max(np.abs(
                        feed[name][0, slot] - frame[key]))))
        outputs = self.session.run(names, feed)
        self.targets.append(outputs[names.index("joint_pos_targets")][0].copy())
        return outputs


class Trace:
    def __init__(self, protocol: dict, dt: float):
        self.protocol, self.dt = protocol, dt
        self.rows: list[list] = []
        self.peak_utilization = np.zeros(29)
        self.saturation_counts = np.zeros(29, dtype=np.int64)
        self.substeps = 0
        self.max_pd_force_error = 0.0
        self.held_state = None

    def observe(self, model, data, step, reference):
        normal, _ = contact_telemetry(model, data)
        feet = data.xpos[[model.body(name).id for name in FEET]]
        self.rows.append([
            round(step * self.dt, 6), *data.qpos[7:19], *reference["q"][:12],
            *data.qpos[:3], *normal, *feet[:, 2],
        ])
        if step == self.protocol["probe_control_step"]:
            self.held_state = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())

    def substep(self, model, data):
        force = data.actuator_force
        limit = model.actuator_forcerange[:, 1]
        utilization = np.abs(force) / limit
        self.peak_utilization = np.maximum(self.peak_utilization, utilization)
        self.saturation_counts += utilization >= self.protocol["torque_saturation_fraction"]
        expected = np.clip(model.actuator_gainprm[:, 0] * (data.ctrl - data.actuator_length)
                           + model.actuator_biasprm[:, 2] * data.actuator_velocity, -limit, limit)
        self.max_pd_force_error = max(self.max_pd_force_error, float(np.max(np.abs(force - expected))))
        self.substeps += 1

    def summarize(self, command: CommandTrace) -> dict:
        values = np.asarray(self.rows)
        commanded = np.asarray(command.targets).reshape(-1, 29)
        actual, requested = values[:, 1:13], values[:, 13:25]
        after_hold = values[:, 0] >= 1
        usable = min(len(commanded), len(actual) - 1)
        command_after_hold = commanded[values[:len(commanded), 0] >= 1, :12]
        return {
            "maxHistoryErrorRad": command.max_history_error,
            "maxFutureInputError": command.max_future_error,
            "substeps": self.substeps,
            "peakTorqueUtilizationPerJoint": self.peak_utilization.tolist(),
            "saturatedSubstepsPerJoint": self.saturation_counts.tolist(),
            "maxPdForceEquationErrorNm": self.max_pd_force_error,
            "requestedRangePerLegJointRad": np.ptp(requested[after_hold], axis=0).tolist() if after_hold.any() else None,
            "commandedRangePerLegJointRad": np.ptp(command_after_hold, axis=0).tolist() if len(command_after_hold) else None,
            "realizedRangePerLegJointRad": np.ptp(actual[after_hold], axis=0).tolist() if after_hold.any() else None,
            "commandVsRequestedRmsePerLegJointRad": np.sqrt(np.mean(
                (commanded[:usable, :12] - requested[:usable]) ** 2, axis=0)).tolist() if usable else None,
            "realizedVsRequestedRmsePerLegJointRad": np.sqrt(np.mean((actual - requested) ** 2, axis=0)).tolist(),
            "minimumNormalForceAfterHoldN": np.min(values[after_hold, 28:30], axis=0).tolist() if after_hold.any() else None,
            "minimumLoadShareAfterHold": np.min(values[after_hold, 28:30] /
                np.maximum(np.sum(values[after_hold, 28:30], axis=1, keepdims=True), 1e-12), axis=0).tolist() if after_hold.any() else None,
            "footHeightExcursionAfterHoldM": np.ptp(values[after_hold, 30:32], axis=0).tolist() if after_hold.any() else None,
            "controlSamplesColumns": ["time", "actual_leg_joints[12]", "reference_leg_joints[12]",
                                       "root_xyz[3]", "normal_force_lr[2]", "foot_z_lr[2]"],
            "controlSamples": np.round(values, self.protocol["sample_precision_decimal_places"]).tolist(),
            "commandLegJoints": np.round(commanded[:, :12], self.protocol["sample_precision_decimal_places"]).tolist(),
            "commandAlignment": "command[k] is issued at sample[k], then held for 20 substeps until sample[k+1].",
        }


def static_probes(model, meta, default, session, instrumented, initializers, state, references, protocol):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], history = state
    mujoco.mj_forward(model, data)
    # The reference torso has a fixed local rotation in the shipped MJCF.
    ref_data = mujoco.MjData(model)
    ref_data.qpos[3:7], ref_data.qpos[7:] = [1, 0, 0, 0], default
    mujoco.mj_kinematics(model, ref_data)
    anchor = xyzw(ref_data.xquat[meta["robot"]["anchor_body_index"] + 1])
    stand = [{"q": default, "v": np.zeros(29), "anchor": anchor} for _ in range(4)]
    baseline = make_inputs(data, meta, stand, history)
    feeds = [("stand", baseline)]
    for slot, offset in enumerate(meta["motion"]["future_step_indices"]):
        feed = {name: value.copy() for name, value in baseline.items()}
        feed["mimic_future_dof_pos"][0, slot, 3] += protocol["knee_probe_rad"]
        feeds.append((f"left_knee_plus_0.1_at_offset_{offset}", feed))
    for label, key, index, delta in (
        ("right_knee_plus_0.1_at_offset_8", "mimic_future_dof_pos", (0, 3, 9), protocol["knee_probe_rad"]),
        ("left_knee_velocity_plus_0.2_at_offset_8", "mimic_future_dof_vel", (0, 3, 3), protocol["velocity_probe_rad_s"]),
    ):
        feed = {name: value.copy() for name, value in baseline.items()}
        feed[key][index] += delta
        feeds.append((label, feed))
    feed = {name: value.copy() for name, value in baseline.items()}
    angle = np.deg2rad(protocol["future_anchor_yaw_probe_deg"]) / 2
    yaw, composed = np.array([np.cos(angle), 0, 0, np.sin(angle)]), np.empty(4)
    mujoco.mju_mulQuat(composed, yaw, anchor[[3, 0, 1, 2]].astype(np.float64))
    feed["mimic_future_anchor_rot"][0, 0] = composed[[1, 2, 3, 0]]
    feeds.append(("future_anchor_yaw_plus_5_degrees", feed))
    step = protocol["probe_control_step"]
    for name, frames in references.items():
        future = [frames[step + offset] for offset in meta["motion"]["future_step_indices"]]
        feeds.append((name + "_swing", make_inputs(data, meta, future, history)))
    translated = [{**frame, "body": np.ones((33, 3)) * protocol["passthrough_translation_m"]} for frame in stand]
    feeds.append(("passthrough_position_only", make_inputs(data, meta, translated, history)))
    results = []
    for name, feed in feeds:
        result = probe(instrumented, initializers, session, feed)
        if not results:
            base = result
        results.append({
            "name": name,
            "changedInputElements": {key: int(np.count_nonzero(feed[key] != baseline[key])) for key in feed},
            "inputMaximumAbsoluteDelta": {key: float(np.max(np.abs(feed[key] - baseline[key]))) for key in feed},
            "targetDeltaPerJointRad": (np.asarray(result["target"]) - base["target"]).tolist(),
            "targetDeltaMaxRad": float(np.max(np.abs(np.asarray(result["target"]) - base["target"]))),
            "rawObservationDeltaMax": float(np.max(np.abs(np.asarray(result["rawObservation"]) - base["rawObservation"]))),
            **result,
        })
    return {"heldTimeS": step * meta["timing"]["control_dt"], "stateQpos": state[0].tolist(),
            "stateQvel": state[1].tolist(), "history": history.tolist(), "probes": results}


def run(assets: Path, sources: Path, output: Path, fetch: bool) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schemaVersion": 1, "kind": "deployment-contract-diagnosis", "identity": protocol["identity"],
        "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "protocolSha256": digest(PROTOCOL), "revision": REVISION,
        "historicalSha256": {name: digest(Path(name)) for name in HISTORICAL},
        "sourceSha256": {name: digest(Path(__file__).with_name(name)) for name in ("diagnose.py", "rollout.py", "locomotion.py")},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("numpy", "mujoco", "onnxruntime", "onnx", "ml_dtypes", "PyYAML")},
                    "provider": "CPUExecutionProvider", "inferenceThreads": 1},
        "cases": [],
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file)

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    try:
        result["sources"] = source_records(sources, fetch)
        result["provenance"] = verify_assets(assets)
        directory = assets / MODEL_DIR
        meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        default = np.asarray(config["robot"]["default_dof_pos"])
        model = load_model(assets, meta, config)
        session = make_session(directory / "compiled_models/unified_pipeline.onnx", meta)
        instrumented, initializers, result["graph"] = inspect_graph(directory / "compiled_models/unified_pipeline.onnx", meta, config)
        result["physicsContract"] = {
            "jointNames": meta["robot"]["joint_names"], "bodyNames": meta["robot"]["body_names"],
            "jointAxes": model.jnt_axis[1:].tolist(), "actuatorJointIds": model.actuator_trnid[:, 0].tolist(),
            "actuatorGear": model.actuator_gear.tolist(), "stiffness": model.actuator_gainprm[:, 0].tolist(),
            "damping": (-model.actuator_biasprm[:, 2]).tolist(), "effortLimits": model.actuator_forcerange[:, 1].tolist(),
            "armature": model.dof_armature[6:].tolist(), "timing": meta["timing"],
            "integrator": int(model.opt.integrator), "gravity": model.opt.gravity.tolist(),
            "floorFriction": model.geom_friction[model.geom("floor").id].tolist(),
            "footFriction": {model.geom(i).name: model.geom_friction[i].tolist()
                             for i in range(model.ngeom) if "foot" in model.geom(i).name},
        }
        references = {}
        for number in (1, 2):
            path = PROTOCOL_PATH if number == 1 else PROTOCOL_PATH.with_name("locomotion-protocol-02.json")
            historical = json.loads(Path(f"public/locomotion/attempt-{number:02d}.json").read_text())
            if digest(path) != historical["protocolSha256"]:
                raise ValueError("Historical locomotion protocol changed.")
            frames, _ = generate_reference(model, meta, default, json.loads(path.read_text()))
            references[f"attempt-{number:02d}"] = frames
        save()
        for name in protocol["cases"]:
            trace = Trace(protocol, meta["timing"]["control_dt"])
            commands = CommandTrace(session, references.get(name), meta)
            case = run_case(model, commands, meta, default, "stand" if name == "stand" else "slow_walk",
                            protocol["seconds"], reference_frames=references.get(name), record_frames=False,
                            wall_limit_seconds=protocol["wall_limit_seconds_per_case"],
                            observe=trace.observe, observe_substep=trace.substep)
            item = {"name": name, "physics": case, "trace": trace.summarize(commands)}
            result["cases"].append(item)
            save()
            if not case["metrics"]["completed"]:
                raise RuntimeError(f"Diagnostic case {name} incomplete; partial evidence saved.")
            if name == "stand":
                result["staticProbes"] = static_probes(model, meta, default, session, instrumented, initializers,
                                                       trace.held_state, references, protocol)
            else:
                historical = json.loads(Path(f"public/locomotion/{name}.json").read_text())
                keys = ("joint_rmse_rad", "mean_body_position_error_m", "min_root_height_m", "max_pelvis_tilt_deg")
                item["historicalMetricAbsoluteDelta"] = {
                    key: abs(case["metrics"][key] - historical["physics"]["metrics"][key]) for key in keys}
                item["historicalRootMaxAbsoluteDeltaM"] = float(np.max(np.abs(
                    np.asarray(trace.rows)[:, 25:28] - [sample["root"] for sample in historical["locomotion"]["samples"]])))
                if max(*item["historicalMetricAbsoluteDelta"].values(), item["historicalRootMaxAbsoluteDeltaM"]) > protocol["historical_metric_tolerance"]:
                    raise ValueError("Diagnostic instrumentation changed a historical trajectory.")
            save()
        if {name: digest(Path(name)) for name in HISTORICAL} != result["historicalSha256"]:
            raise ValueError("Historical evidence was modified.")
        result.update(status="complete", finishedAt=datetime.now(timezone.utc).isoformat())
        save()
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--sources", type=Path, default=Path(".research-cache/contract-sources"))
    parser.add_argument("--output", type=Path, default=Path("public/diagnostics/contract-audit-v1.json"))
    parser.add_argument("--fetch-sources", action="store_true", help="Fetch only pinned source text, never execute it.")
    args = parser.parse_args()
    run(args.assets, args.sources, args.output, args.fetch_sources)
