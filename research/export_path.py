"""Export the measured baseline/feedback pair without changing any recorded trial."""

import argparse
import json
from pathlib import Path

from path_tracking import OUTPUT, digest


def export(directory):
    files = {name: directory / (name + ".json") for name in ("diagnosis", "invariance-resolution", "feedback")}
    records = {name: json.loads(path.read_text()) for name, path in files.items()}
    diagnosis, resolution, feedback = (records[name] for name in files)
    if (any(r["status"] != "complete" for r in records.values()) or not resolution["passed"]
        or feedback["diagnosisSha256"] != digest(files["diagnosis"])
        or feedback["resolutionSha256"] != digest(files["invariance-resolution"])
        or len(diagnosis["episodes"]) != 2 or len(feedback["episodes"]) != 1):
        raise ValueError("Expected complete linked diagnosis, resolution and one feedback trial.")
    reference_path = Path(feedback["protocol"]["reference_artifact"])
    if digest(reference_path) != feedback["referenceArtifactSha256"]:
        raise ValueError("Measured reference lineage changed.")
    reference = json.loads(reference_path.read_text())
    cases = []
    for episode in (diagnosis["episodes"][0], feedback["episodes"][0]):
        if not episode["physics"]["metrics"]["completed"]:
            raise ValueError("This export requires complete trials; do not conceal truncation.")
        cases.append({**episode["physics"],
                      "locomotion": {key: episode["locomotion"][key] for key in ("passed", "gates", "metrics")},
                      "pathMetrics": episode["relativeMetrics"]})
    if [c["name"] for c in cases] != ["baseline", "heading_feedback"]:
        raise ValueError("Unexpected comparison case identity.")
    replay = {
        "schemaVersion": 1, "kind": "recorded-pretrained-policy-physics", "experimentId": feedback["id"],
        "createdAt": feedback["createdAt"], "policy": "NVIDIA G1 tracker / new external heading feedback comparison",
        "reference": "Identical CMU07/04 fitted reference, moving initialization and physics. Reference skeleton is never aligned to hide drift.",
        "provenance": feedback["provenance"], "bodyNames": reference["bodyNames"], "parents": reference["parents"],
        "cases": cases, "sourceArtifacts": {path.name: digest(path) for path in files.values()},
        "referenceArtifactSha256": digest(reference_path), "protocol": feedback["protocol"],
        "limitations": [
            "Heading feedback is new external control, not an achievement of the unmodified pretrained policy.",
            "One frozen feedback trial; no gain search, start-from-rest result, generalization or SOTA claim.",
            "All original physical gates remain. Both cases fail lateral drift, root tracking and body tracking.",
            "Lateral/root tracking improves, but along-track lag and left-foot slip worsen.",
            "Strict full-joint yaw-equivariance diagnostic failed; it remains failed, alongside the independent one-step resolution.",
        ],
    }
    output = directory / "physics-replay.json"
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(replay, separators=(",", ":"), allow_nan=False) + "\n")
    return replay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    replay = export(args.input_dir)
    print(json.dumps({"cases": [c["name"] for c in replay["cases"]], "sources": replay["sourceArtifacts"]}, indent=2))
