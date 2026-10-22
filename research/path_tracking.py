"""Audit CMU path tracking, then test one frozen causal torso-yaw controller."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from cmu_contact import source_targets
from diagnose import expected_observation
from fetch_assets import MODEL_DIR
from locomotion import LocomotionObserver
from rollout import load_model, make_inputs, make_session, run_case, verify_assets, xyzw

PROTOCOL = Path(__file__).with_name("path-tracking-protocol.json")
OUTPUT = Path("public/path-tracking-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(qpos):
    return Rotation.from_quat(np.asarray(qpos)[[4, 5, 6, 3]])


def yaw(qpos):
    matrix = rotation(qpos).as_matrix()
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def transform_state(position, velocity, world_rotation, translation):
    q, v = position.copy(), velocity.copy()
    q[:3] = world_rotation.apply(q[:3]) + translation
    q[3:7] = (world_rotation * rotation(q)).as_quat()[[3, 0, 1, 2]]
    v[:3] = world_rotation.apply(v[:3])
    # The free-joint angular component is body-local, unlike its linear part.
    return q, v


def reference_frames(model, meta, evidence, world_rotation=None, translation=None):
    frames, data = [], mujoco.MjData(model)
    for p, v, body, stance in zip(evidence["qpos"], evidence["qvel"], evidence["body"], evidence["stance"]):
        p, v = np.asarray(p), np.asarray(v)
        if world_rotation is not None:
            p, v = transform_state(p, v, world_rotation, translation)
            body = world_rotation.apply(body) + translation
        data.qpos[:], data.qvel[:] = p, v
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[1:], body, atol=1e-11, rtol=0)
        frames.append({"qpos": p, "qvel": v, "q": p[7:].astype(np.float32), "v": v[6:].astype(np.float32),
                       "anchor": xyzw(data.xquat[meta["robot"]["anchor_body_index"] + 1]),
                       "root": p[:3].copy(), "body": np.asarray(body), "stance": stance})
    return frames


def path_basis(references, last):
    delta = references[last]["root"][:2] - references[0]["root"][:2]
    if np.linalg.norm(delta) < 0.1:
        raise ValueError("This fixed path experiment needs nonzero reference displacement.")
    along = delta / np.linalg.norm(delta)
    return along, np.array([-along[1], along[0]])


def relative_metrics(roots, targets):
    roots, targets = np.asarray(roots), np.asarray(targets)
    delta = targets[-1, :2] - targets[0, :2]
    along = delta / np.linalg.norm(delta)
    cross = np.array([-along[1], along[0]])
    errors = roots[:, :2] - targets[:, :2]
    displacement_error = (roots[-1] - roots[0] - targets[-1] + targets[0])[:2]
    return {
        "root_xy_rmse_m": float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
        "cross_track_rmse_m": float(np.sqrt(np.mean((errors @ cross) ** 2))),
        "final_cross_track_error_m": float(errors[-1] @ cross),
        "final_along_track_error_m": float(errors[-1] @ along),
        "final_relative_lateral_displacement_error_m": float(displacement_error @ cross),
        "reference_world_y_displacement_m": float(targets[-1, 1] - targets[0, 1]),
    }


class HeadingFeedback:
    def __init__(self, settings, references, last, dt):
        self.settings, self.dt = settings, dt
        self.along, self.cross = path_basis(references, last)
        self.angle = 0.

    def update(self, root, velocity, reference):
        s = self.settings
        error = float((reference["root"][:2] - root[:2]) @ self.cross)
        velocity_error = float((reference["qvel"][:2] - velocity[:2]) @ self.cross)
        forward = max(float(reference["qvel"][:2] @ self.along), s["minimum_forward_speed_m_s"])
        raw = float(np.arctan2(s["position_gain_per_second"] * error + s["velocity_gain"] * velocity_error, forward))
        bounded = float(np.clip(raw, -np.radians(s["maximum_yaw_degrees"]), np.radians(s["maximum_yaw_degrees"])))
        filtered = self.angle + (1 - np.exp(-self.dt / s["smoothing_time_s"])) * (bounded - self.angle)
        maximum_step = np.radians(s["maximum_yaw_rate_degrees_s"]) * self.dt
        self.angle += float(np.clip(filtered - self.angle, -maximum_step, maximum_step))
        return {"angle_rad": self.angle, "raw_angle_rad": raw, "cross_track_error_m": error,
                "lateral_velocity_error_m_s": velocity_error, "reference_forward_speed_m_s": forward}

    @staticmethod
    def apply(feed, angle):
        if angle == 0:
            return feed
        result = dict(feed)
        turns = Rotation.from_rotvec([0., 0., angle]) * Rotation.from_quat(feed["mimic_future_anchor_rot"][0])
        result["mimic_future_anchor_rot"] = turns.as_quat().astype(np.float32)[None]
        return result


class PolicyTrace:
    def __init__(self, session, references, feedback=None):
        self.session, self.references, self.feedback = session, references, feedback
        self.state = None
        self.commands, self.corrections = [], []

    def run(self, names, feed):
        step = len(self.commands)
        if self.state is None or self.state[0] != step:
            raise ValueError("Policy correction must use the current causal observed state.")
        if self.feedback is not None:
            _, root, velocity = self.state
            correction = self.feedback.update(root, velocity, self.references[step])
            feed = self.feedback.apply(feed, correction["angle_rad"])
            self.corrections.append(correction)
        outputs = self.session.run(names, feed)
        self.commands.append(outputs[names.index("joint_pos_targets")][0].copy())
        return outputs


class StateTrace:
    def __init__(self, protocol, dt, policy):
        self.observer, self.policy = LocomotionObserver(protocol, dt), policy
        self.states = []

    def __call__(self, model, data, step, reference):
        self.observer(model, data, step, reference)
        mujoco.mj_subtreeVel(model, data)
        self.states.append({"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                            "com": data.subtree_com[1].tolist(), "comVelocity": data.subtree_linvel[1].tolist(),
                            "rootYaw": yaw(data.qpos),
                            "torsoYaw": float(np.arctan2(data.xmat[model.body("torso_link").id, 3],
                                                         data.xmat[model.body("torso_link").id, 0]))})
        self.policy.state = step, data.qpos[:3].copy(), data.qvel[:3].copy()


def episode(model, session, meta, default, reference, old, protocol, name, corrected=False):
    feedback = HeadingFeedback(protocol["feedback"], reference, round(protocol["seconds"] / .02), .02) if corrected else None
    policy = PolicyTrace(session, reference, feedback)
    trace = StateTrace(old["physicalProtocol"], .02, policy)
    physical = run_case(model, policy, meta, default, name, protocol["seconds"], reference_frames=reference,
                        initial_state=(reference[0]["qpos"], reference[0]["qvel"]), observe=trace,
                        wall_limit_seconds=protocol["wall_limit_seconds_per_case"])
    locomotion = trace.observer.summarize(physical["metrics"])
    return {"physics": physical, "locomotion": locomotion, "states": trace.states,
            "relativeMetrics": relative_metrics([s["root"] for s in locomotion["samples"]],
                                                [s["referenceRoot"] for s in locomotion["samples"]]),
            "commands": np.asarray(policy.commands).tolist(), "corrections": policy.corrections}


def reference_diagnosis(model, default, frames, old):
    q = np.asarray([f["qpos"] for f in frames])
    v = np.asarray([f["qvel"] for f in frames])
    angular_error = 0.
    for i in range(len(q)):
        a, b = max(0, i - 1), min(len(q) - 1, i + 1)
        angular = (rotation(q[a]).inv() * rotation(q[b])).as_rotvec() / ((b - a) * .02)
        angular_error = max(angular_error, float(np.max(np.abs(angular - v[i, 3:6]))))
    targets, _, _, _ = source_targets(model, default, old["protocol"])
    source_root = np.asarray([target.root for target in targets])
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q[0], v[0]
    mujoco.mj_forward(model, data)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac, None, 1)
    root_world_velocity_error = float(np.max(np.abs(jac @ data.qvel - v[0, :3])))
    local_angular_error = float(np.max(np.abs(data.xmat[1].reshape(3, 3).T @ data.cvel[1, :3] - v[0, 3:6])))
    active = [model.joint(i + 1).name for i in range(29)
              if min(q[0, i + 7] - model.jnt_range[i + 1, 0], model.jnt_range[i + 1, 1] - q[0, i + 7]) < 1e-5]
    rows = []
    for i, frame in enumerate(frames[:187]):
        data.qpos[:], data.qvel[:] = frame["qpos"], frame["qvel"]
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        rows.append({"time": i * .02, "root": frame["root"].tolist(), "velocity": frame["qvel"][:3].tolist(),
                     "rootYaw": yaw(frame["qpos"]), "torsoYaw": float(Rotation.from_quat(frame["anchor"]).as_euler("ZYX")[0]),
                     "com": data.subtree_com[1].tolist(), "comVelocity": data.subtree_linvel[1].tolist()})
    return {
        "translationDerivativeMaxError": float(np.max(np.abs(np.gradient(q[:, :3], .02, axis=0) - v[:, :3]))),
        "jointDerivativeMaxError": float(np.max(np.abs(np.gradient(q[:, 7:], .02, axis=0) - v[:, 6:]))),
        "angularDerivativeMaxError": angular_error,
        "rootWorldVelocityJacobianMaxError": root_world_velocity_error,
        "bodyLocalAngularVelocityMaxError": local_angular_error,
        "initialFittedRootVelocity": v[0, :3].tolist(),
        "initialSourceRootVelocity": ((source_root[1] - source_root[0]) / .02).tolist(),
        "initialFitRootAdjustments": (q[:2, :3] - source_root[:2]).tolist(),
        "initialActiveJointLimits": active, "referenceSamples": rows,
        "interpretation": "The initial fitted derivative includes release of a 4.57 mm root-Y adaptation at an active left-ankle-roll bound. It is not the raw source velocity, but is consistent with the saved fitted trajectory. This alone does not explain sustained drift or justify replacing velocity independently of the reference.",
    }


def graph_invariance(model, session, meta, baseline, references, transformed, world_rotation, translation, indices):
    a, b = mujoco.MjData(model), mujoco.MjData(model)
    reports = []
    for step in indices:
        state = baseline["states"][step]
        p, v = np.asarray(state["qpos"]), np.asarray(state["qvel"])
        a.qpos[:], a.qvel[:] = p, v
        b.qpos[:], b.qvel[:] = transform_state(p, v, world_rotation, translation)
        mujoco.mj_forward(model, a)
        mujoco.mj_forward(model, b)
        previous = np.asarray(baseline["commands"][step - 1]) if step else np.zeros(29)
        offsets = meta["motion"]["future_step_indices"]
        feeds = [make_inputs(data, meta, [ref[step + o] for o in offsets], previous)
                 for data, ref in ((a, references), (b, transformed))]
        outputs = [session.run(["joint_pos_targets"], feed)[0] for feed in feeds]
        reports.append({"step": step,
                        "observationMaxError": float(np.max(np.abs(expected_observation(feeds[0]) - expected_observation(feeds[1])))),
                        "policyTargetMaxErrorRad": float(np.max(np.abs(outputs[0] - outputs[1])))})
    return {"staticPolicyCalls": len(reports) * 2, "samples": reports}


def diagnose(model, session, meta, default, frames, old, protocol, record_episode):
    reference = reference_diagnosis(model, default, frames, old)
    baseline = episode(model, session, meta, default, frames, old, protocol, "baseline")
    record_episode(baseline)
    turn = Rotation.from_euler("z", protocol["common_transform"]["yaw_degrees"], degrees=True)
    shift = np.asarray(protocol["common_transform"]["translation_m"])
    rotated = reference_frames(model, meta, old["referenceEvidence"], turn, shift)
    transformed = episode(model, session, meta, default, rotated, old, protocol, "baseline_common_transform")
    record_episode(transformed)
    roots = np.asarray([s["qpos"][:3] for s in baseline["states"]])
    inverse_roots = turn.inv().apply(np.asarray([s["qpos"][:3] for s in transformed["states"]]) - shift)
    expected = np.asarray([s["root"] for s in old["locomotion"]["samples"]])
    if roots.shape != expected.shape or inverse_roots.shape != roots.shape:
        raise ValueError("Incomplete diagnostic rollout; retain the execution error, do not compare truncated trajectories.")
    graph = graph_invariance(model, session, meta, baseline, frames, rotated, turn, shift, protocol["probe_indices"])
    invariance = {
        "baselineRootReproductionMaxM": float(np.max(np.abs(roots - expected))),
        "commonTransformRootMaxM": float(np.max(np.linalg.norm(inverse_roots - roots, axis=1))),
        "commonTransformJointMaxRad": float(np.max(np.abs(np.asarray([s["qpos"][7:] for s in transformed["states"]])
                                                       - np.asarray([s["qpos"][7:] for s in baseline["states"]])))),
        "bodyTrackingMeanDifferenceM": abs(baseline["physics"]["metrics"]["mean_body_position_error_m"]
                                           - transformed["physics"]["metrics"]["mean_body_position_error_m"]),
    }
    limits = protocol["diagnostic_limits"]
    checks = {
        "baseline_reproduces": invariance["baselineRootReproductionMaxM"] <= limits["baseline_root_reproduction_m"],
        "derivatives_and_frames": max(reference[key] for key in reference if key.endswith("MaxError")) <= limits["derivative_consistency"],
        "graph_observation_invariance": max(p["observationMaxError"] for p in graph["samples"]) <= limits["graph_observation_invariance"],
        "graph_action_invariance": max(p["policyTargetMaxErrorRad"] for p in graph["samples"]) <= limits["graph_target_invariance_rad"],
        "physical_invariance": invariance["commonTransformRootMaxM"] <= limits["trajectory_invariance_m"]
                               and invariance["commonTransformJointMaxRad"] <= limits["joint_trajectory_invariance_rad"],
    }
    return {"checks": checks, "passed": all(checks.values()), "reference": reference, "invariance": invariance,
            "graph": graph, "episodes": [baseline, transformed]}


def run(stage, assets, output):
    protocol = json.loads(PROTOCOL.read_text())
    source_path = Path(protocol["reference_artifact"])
    if digest(source_path) != protocol["reference_sha256"]:
        raise ValueError("The immutable reference source changed.")
    old = json.loads(source_path.read_text())
    result = {"schemaVersion": 1, "kind": "cmu-path-tracking-experiment", "id": protocol["id"], "stage": stage,
              "status": "running", "createdAt": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "protocolSha256": digest(PROTOCOL), "referenceArtifactSha256": digest(source_path),
              "sourceSha256": {name: digest(Path(__file__).with_name(name)) for name in
                               ("path_tracking.py", "rollout.py", "locomotion.py", "cmu_contact.py")},
              "physicalCriteria": old["physicalProtocol"]["criteria"],
              "historicalSha256": {p.as_posix(): digest(p) for p in Path("public").rglob("*.json") if OUTPUT not in p.parents},
              "episodes": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file)
    def save():
        temporary = output.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(output)
    try:
        result["provenance"] = verify_assets(assets)
        directory = assets / MODEL_DIR
        meta = yaml.safe_load((directory / "compiled_models/unified_pipeline.yaml").read_text())
        config = yaml.safe_load((directory / "resolved_configs_inference.yaml").read_text())
        default = np.asarray(config["robot"]["default_dof_pos"])
        model = load_model(assets, meta, config)
        frames = reference_frames(model, meta, old["referenceEvidence"])
        session = make_session(directory / "compiled_models/unified_pipeline.onnx", meta)
        def record_episode(value):
            result["episodes"].append(value)
            save()
        if stage == "diagnose":
            result.update(diagnose(model, session, meta, default, frames, old, protocol, record_episode))
        else:
            diagnosis_path = output.parent / "diagnosis.json"
            diagnosis = json.loads(diagnosis_path.read_text())
            if (diagnosis["status"] != "complete" or not diagnosis["passed"]
                or diagnosis["protocolSha256"] != result["protocolSha256"]):
                raise ValueError("Feedback requires the completed matching invariant diagnosis first.")
            result["diagnosisSha256"] = digest(diagnosis_path)
            measured = episode(model, session, meta, default, frames, old, protocol, "heading_feedback", corrected=True)
            result["episodes"] = [measured]
            result["passed"] = measured["locomotion"]["passed"]
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(json.dumps({"stage": stage, "passed": result["passed"], "checks": result.get("checks"),
                      "invariance": result.get("invariance"),
                      "episodes": [{"name": e["physics"]["name"], "metrics": e["physics"]["metrics"],
                                    "relative": e["relativeMetrics"], "gates": e["locomotion"]["gates"]}
                                   for e in result["episodes"]]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnose", "feedback"), required=True)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.stage, args.assets, args.output_dir / ("diagnosis.json" if args.stage == "diagnose" else "feedback.json"))
    if not result["passed"]:
        raise SystemExit("Fixed experiment checks failed; measured evidence retained.")
