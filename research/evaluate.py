"""Execute the fixed, matched 12-episode CPU comparison; never overwrite the replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np
import yaml

from fetch_assets import MODEL_DIR
from rollout import Push, load_model, make_session, run_case, verify_assets

PROTOCOL_PATH = Path(__file__).with_name("comparison-protocol.json")


def make_pushes(protocol: dict, seed: int | None) -> tuple[Push, ...]:
    if seed is None:
        return ()
    rng = np.random.Generator(np.random.PCG64(seed))
    dt = protocol["control_dt"]
    pushes = []
    for start, end in protocol["push_intervals_s"]:
        angle = rng.uniform(0, 2 * math.pi)
        force = protocol["push_force_n"]
        pushes.append(Push(round(start / dt), round(end / dt),
                           (float(force * math.cos(angle)), float(force * math.sin(angle)), 0.0)))
    return tuple(pushes)


def aggregate(episodes: list[dict], expected_per_controller: int) -> list[dict]:
    summaries = []
    for controller in ("onnx", "direct_pd"):
        selected = [episode for episode in episodes if episode["controller"] == controller]
        metrics = [episode["metrics"] for episode in selected]
        count = len(metrics)
        samples = sum(item["error_samples"] for item in metrics)
        pooled = math.sqrt(sum(item["joint_rmse_rad"] ** 2 * item["error_samples"] for item in metrics) / samples) if samples else None
        all_complete = count == expected_per_controller and all(item["completed"] for item in metrics)
        falls = sum(item["fell"] for item in metrics)
        failures = sum(not item["completed"] or item["fell"] for item in metrics)
        summaries.append({
            "controller": controller,
            "expected_episodes": expected_per_controller,
            "attempted_episodes": count,
            "completed_episodes": sum(item["completed"] for item in metrics),
            "successes": count - failures,
            "falls": falls,
            "failures": failures,
            "early_terminations": sum(not item["completed"] for item in metrics),
            "fall_rate": falls / count if count else None,
            "failure_rate": failures / count if count else None,
            "observed_error_samples": samples,
            "observed_joint_rmse_rad": pooled,
            "full_horizon_joint_rmse_rad": pooled if all_complete else None,
            "observed_mean_body_error_m": sum(item["mean_body_position_error_m"] * item["error_samples"] for item in metrics) / samples if samples else None,
            "simulated_seconds": sum(item["simulated_seconds"] for item in metrics),
            "wall_seconds": sum(item["wall_seconds"] for item in metrics),
        })
    return summaries


def evaluate(assets: Path, output: Path) -> dict:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["id"] != "g1-synthetic-matched-v1" or protocol["seconds_per_episode"] != 5 or protocol["wall_limit_seconds_per_episode"] != 90:
        raise ValueError("This runner implements only the fixed bounded v1 protocol.")
    provenance = verify_assets(assets)
    model_dir = assets / MODEL_DIR
    meta = yaml.safe_load((model_dir / "compiled_models/unified_pipeline.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((model_dir / "resolved_configs_inference.yaml").read_text(encoding="utf-8"))
    for key in ("physics_dt", "control_dt"):
        if not math.isclose(meta["timing"][key], protocol[key]):
            raise ValueError(f"Protocol timing mismatch: {key}")
    if meta["robot"]["body_names"][0] != protocol["push_body"]:
        raise ValueError("Disturbances must act on the floating-base pelvis.")
    model = load_model(assets, meta, config)
    session = make_session(model_dir / "compiled_models/unified_pipeline.onnx", meta)
    default = np.asarray(config["robot"]["default_dof_pos"], dtype=np.float64)
    result = {
        "schemaVersion": 1,
        "kind": "matched-controller-comparison",
        "status": "running",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocolSha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "provenance": provenance,
        "sourceSha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("evaluate.py", "rollout.py")},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "onnxruntime", "PyYAML")},
            "provider": "CPUExecutionProvider", "inference_threads": 1,
        },
        "episodes": [],
        "summary": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    print(f"Fixed protocol SHA-256: {result['protocolSha256']}", flush=True)
    pair_index = 0
    for reference in protocol["references"]:
        for condition in protocol["conditions"]:
            pushes = make_pushes(protocol, condition["seed"])
            controllers = protocol["controllers"] if pair_index % 2 == 0 else list(reversed(protocol["controllers"]))
            for controller in controllers:
                episode = run_case(model, session if controller == "onnx" else None, meta, default, reference,
                                   protocol["seconds_per_episode"], controller=controller, pushes=pushes,
                                   record_frames=False, wall_limit_seconds=protocol["wall_limit_seconds_per_episode"])
                del episode["frames"]
                episode.update(condition=condition["name"], seed=condition["seed"], pushes=[asdict(push) for push in pushes])
                result["episodes"].append(episode)
                result["summary"] = aggregate(result["episodes"], 6)
                save()
            pair_index += 1
    result["status"] = "complete"
    result["finishedAt"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".research-cache/protomotions"))
    parser.add_argument("--output", type=Path, default=Path("public/physics-comparison.json"))
    args = parser.parse_args()
    result = evaluate(args.assets, args.output)
    if any(item["early_terminations"] for item in result["summary"]):
        raise SystemExit("Comparison saved with early terminations; inspect every episode, not only full-horizon errors.")
