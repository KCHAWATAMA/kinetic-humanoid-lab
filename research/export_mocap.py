"""Project immutable measured CMU experiments into replay and execution-ledger files."""

import argparse
import hashlib
import json
from pathlib import Path

HISTORICAL = [
    "public/physics-replay.json", "public/physics-comparison.json",
    "public/locomotion/attempt-01.json", "public/locomotion/attempt-02.json", "public/locomotion/summary.json",
    "public/diagnostics/contract-audit-v1.json", "public/mocap/g1-cmu-07-04-v1.json",
    "public/mocap/interrupted-preflight.json", "public/mocap/execution-ledger.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(input_dir, output_dir):
    records = [(path, json.loads(path.read_text())) for path in sorted(input_dir.glob("attempt-[0-9][0-9].json"))]
    completed = [(path, value) for path, value in records if value["status"] == "complete" and value["physicsExecuted"]]
    if len(completed) != 1:
        raise ValueError("This checkpoint expects exactly one completed physical CMU trial.")
    path, measured = completed[0]
    if (not measured["reference"]["passed"] or not measured["physics"]["metrics"]["completed"]
        or len(measured["physics"]["frames"]) < 2 or len(measured["bodyNames"]) != 33
        or measured["locomotion"]["passed"] != all(measured["locomotion"]["gates"].values())):
        raise ValueError("Incomplete or inconsistent measured CMU result.")
    replay = {
        "schemaVersion": 1, "kind": "recorded-pretrained-policy-physics", "createdAt": measured["createdAt"],
        "policy": "NVIDIA G1 BONES-SEED deployment tracker",
        "reference": "CMU 07/04, original timing, constrained contact-aware G1 retarget. Moving reference initialization.",
        "experimentId": measured["identity"], "sourceArtifact": path.name, "sourceArtifactSha256": digest(path),
        "provenance": measured["modelProvenance"], "motionProvenance": measured["motionProvenance"],
        "bodyNames": measured["bodyNames"], "parents": measured["parents"], "cases": [measured["physics"]],
        "locomotion": {key: measured["locomotion"][key] for key in ("passed", "gates", "metrics")},
        "limitations": [
            "Actual simulated steps, not reference animation or live browser inference.",
            "Moving reference pose/velocity initialized once; not a start-from-rest controller.",
            "The unchanged trajectory-success gates failed despite verified physical steps.",
            "One short licensed-source trial; no generalization, benchmark superiority or SOTA claim.",
        ],
    }
    ledger = {
        "schemaVersion": 1, "kind": "cmu-contact-execution-ledger",
        "historicalBaseCommit": "b5883f0c48d742a77a346101b61f08d28bfcb308",
        "historicalSha256": {name: digest(Path(name)) for name in HISTORICAL},
        "physicalTrials": sum(bool(value["physicsExecuted"]) for _, value in records),
        "executions": [{
            "artifact": p.name, "sha256": digest(p), "identity": value["identity"], "createdAt": value["createdAt"],
            "status": value["status"], "protocolSha256": value["protocolSha256"],
            "referencePassed": value["reference"]["passed"], "physicsExecuted": value["physicsExecuted"],
            "referenceError": value["reference"].get("error"),
            "failedReferenceGates": [key for key, passed in value["reference"].get("gates", {}).items() if not passed],
            "policyCalls": value.get("physics", {}).get("metrics", {}).get("controller_calls", 0),
            "physicsSteps": value.get("physics", {}).get("metrics", {}).get("physics_steps", 0),
        } for p, value in records],
        "engineeringChanges": {
            "g1-cmu-contact-v2": "Replace marker-height cutoff and flat feet; finite-difference-verified constrained IK. First fit reached small residuals but hit SLSQP iteration limit.",
            "g1-cmu-contact-v2.1": "Jacobian-scaled bounded least-squares initialization before constrained refinement. All poses fit; knee speed exceeded the unchanged 8 rad/s gate.",
            "g1-cmu-contact-v2.2": "Enforce the existing joint-speed budget as consecutive-pose bounds inside both warm and cold solves. Reference passes; one actual physical trial retained, including failed world-space tracking.",
        },
        "interpretation": "Two reference-only rejections followed by one actual physical outcome. No physical success threshold changed; no policy parameter sweep.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"physics-replay.json": replay, "execution-ledger.json": ledger}
    if any((output_dir / name).exists() for name in outputs):
        raise FileExistsError("Refusing to overwrite an existing exported checkpoint.")
    for name, value in outputs.items():
        with (output_dir / name).open("x", encoding="utf-8") as file:
            file.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    return replay, ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("public/mocap-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("public/mocap-v2"))
    args = parser.parse_args()
    _, ledger = export(args.input_dir, args.output_dir)
    print(json.dumps(ledger, indent=2))
