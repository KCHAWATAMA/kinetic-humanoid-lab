"""Numeric-only reader for the selected CMU 07/04 ASF/AMC motion.

CMU's FAQ defines metres = file_length * 0.0254 / ASF length unit.
ASF row-vector C^-1 M C becomes C M C^-1 for our column vectors.
No motion code, pickle, crawling, or other clips are used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from fetch_assets import download

CACHE = Path(".research-cache/cmu-07-04")
ASSETS = {
    "07.asf": {"url": "http://mocap.cs.cmu.edu/subjects/07/07.asf", "maximum_bytes": 64_000,
               "sha256": "c29414591d76a34885bff0b04a2666bf9659f869751a2549591c55277cf95b04"},
    "07_04.amc": {"url": "http://mocap.cs.cmu.edu/subjects/07/07_04.amc", "maximum_bytes": 4_000_000,
                  "sha256": "c37994b8eefa0cf30d1822db05e698f7ce6098400b462cc0fadd185cafa3301e"},
}
ATTRIBUTION = ("The data used in this project was obtained from mocap.cs.cmu.edu. "
               "The database was created with funding from NSF EIA-0196217.")
FORMAT_SOURCE = "https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html"
FLOAT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def numeric(tokens: list[str]) -> np.ndarray:
    if not tokens or any(FLOAT.fullmatch(token) is None for token in tokens):
        raise ValueError(f"Invalid numeric tokens: {tokens[:8]}")
    result = np.asarray([float(token) for token in tokens])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite numeric data.")
    return result


def rotation(values_degrees: np.ndarray, order: list[str]) -> np.ndarray:
    result = np.eye(3)
    for value, axis in zip(values_degrees, order, strict=True):
        angle = np.deg2rad(value)
        c, s = np.cos(angle), np.sin(angle)
        if axis.lower().removeprefix("r") == "x":
            matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis.lower().removeprefix("r") == "y":
            matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis.lower().removeprefix("r") == "z":
            matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Unsupported axis: {axis}")
        result = matrix @ result
    return result


@dataclass
class Bone:
    name: str
    direction: np.ndarray
    length: float
    axes: np.ndarray
    channels: list[str]
    limits: np.ndarray


@dataclass
class Skeleton:
    bones: dict[str, Bone]
    parents: dict[str, str]
    order: list[str]
    scale: float


def meaningful_lines(text: str) -> list[str]:
    return [line.strip() for raw in text.splitlines() if (line := raw.split("#", 1)[0]).strip()]


def parse_asf(text: str) -> Skeleton:
    sections: dict[str, list[str]] = {}
    active = None
    for line in meaningful_lines(text):
        if line.startswith(":"):
            fields = line[1:].split(maxsplit=1)
            active = fields[0]
            if active in sections:
                raise ValueError(f"Duplicate ASF section: {active}")
            sections[active] = fields[1:] if len(fields) == 2 else []
        elif active is not None:
            sections[active].append(line)
    units = {line.split()[0]: line.split()[1:] for line in sections["units"]}
    if units.get("angle") != ["deg"]:
        raise ValueError("Only degree ASF data is supported.")
    length_unit = numeric(units["length"])
    if len(length_unit) != 1 or length_unit[0] <= 0:
        raise ValueError("Invalid ASF length unit.")
    root = {line.split()[0]: line.split()[1:] for line in sections["root"]}
    if (root.get("order") != ["TX", "TY", "TZ", "RX", "RY", "RZ"] or root.get("axis") != ["XYZ"]
        or np.any(numeric(root["position"]) != 0) or np.any(numeric(root["orientation"]) != 0)):
        raise ValueError("Unsupported ASF root contract; expected selected CMU zero-origin XYZ convention.")
    bones = {}
    block = None
    for line in sections["bonedata"]:
        if line == "begin":
            if block is not None:
                raise ValueError("Nested bone block.")
            block = {}
        elif line == "end":
            if block is None:
                raise ValueError("Unmatched bone end.")
            name = block["name"][0]
            channels = block.get("dof", [])
            if name in bones or name == "root" or any(c not in ("rx", "ry", "rz") for c in channels) or len(set(channels)) != len(channels):
                raise ValueError(f"Invalid/duplicate bone: {name}")
            direction, length = numeric(block["direction"]), numeric(block["length"])
            axes = block["axis"]
            if len(direction) != 3 or not np.isclose(np.linalg.norm(direction), 1, atol=1e-4) or len(length) != 1 or length[0] <= 0 or len(axes) != 4 or axes[3] != "XYZ":
                raise ValueError(f"Invalid bone geometry: {name}")
            limits = numeric(block["limits"]).reshape(-1, 2) if channels else np.empty((0, 2))
            if len(limits) != len(channels) or np.any(limits[:, 0] > limits[:, 1]):
                raise ValueError(f"Invalid bone limits: {name}")
            bones[name] = Bone(name, direction, float(length[0]), rotation(numeric(axes[:3]), list("XYZ")), channels, limits)
            block = None
        elif block is not None:
            if line.startswith("("):
                block["limits"].extend(line.replace("(", "").replace(")", "").split())
            else:
                key, *values = line.replace("(", "").replace(")", "").split()
                if key in block:
                    raise ValueError(f"Duplicate bone property: {key}")
                block[key] = values
        else:
            raise ValueError("Bone property outside block.")
    if block is not None or not bones:
        raise ValueError("Incomplete skeleton.")
    parents = {}
    for line in sections["hierarchy"]:
        if line in ("begin", "end"):
            continue
        parent, *children = line.split()
        if parent != "root" and parent not in bones:
            raise ValueError(f"Unknown parent: {parent}")
        for child in children:
            if child not in bones or child in parents:
                raise ValueError(f"Invalid child: {child}")
            parents[child] = parent
    order, remaining = [], set(bones)
    while remaining:
        ready = sorted(name for name in remaining if parents.get(name) == "root" or parents.get(name) in order)
        if not ready:
            raise ValueError("Disconnected or cyclic hierarchy.")
        order.extend(ready)
        remaining.difference_update(ready)
    return Skeleton(bones, parents, order, float(0.0254 / length_unit[0]))


def parse_amc(text: str, skeleton: Skeleton) -> list[dict[str, np.ndarray]]:
    expected = {"root": 6, **{name: len(bone.channels) for name, bone in skeleton.bones.items() if bone.channels}}
    frames, current, directives = [], None, set()
    for line in meaningful_lines(text):
        if line.startswith(":"):
            if current is not None:
                raise ValueError("AMC directive inside motion.")
            directives.add(line)
        elif line.isdecimal():
            if int(line) != len(frames) + 1 or len(frames) >= 20_000:
                raise ValueError("AMC frames must be consecutive from 1 and bounded.")
            current = {}
            frames.append(current)
        else:
            name, *tokens = line.split()
            if current is None or name not in expected or name in current:
                raise ValueError(f"Unexpected AMC channel: {name}")
            values = numeric(tokens)
            if len(values) != expected[name]:
                raise ValueError(f"AMC channel count mismatch: {name}")
            current[name] = values
    if not {":FULLY-SPECIFIED", ":DEGREES"}.issubset(directives) or len(frames) < 2:
        raise ValueError("Expected fully specified degree AMC motion.")
    if any(set(frame) != set(expected) for frame in frames):
        raise ValueError("Missing AMC channels.")
    return frames


def forward(skeleton: Skeleton, frame: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    positions = {"root": frame["root"][:3] * skeleton.scale}
    rotations = {"root": rotation(frame["root"][3:], list("XYZ"))}
    for name in skeleton.order:
        bone, parent = skeleton.bones[name], skeleton.parents[name]
        motion = rotation(frame.get(name, np.empty(0)), bone.channels)
        rotations[name] = rotations[parent] @ bone.axes @ motion @ bone.axes.T
        positions[name] = positions[parent] + rotations[name] @ (bone.direction * bone.length * skeleton.scale)
    return positions, rotations


def fetch(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, spec in ASSETS.items():
        path = destination / name
        payload = path.read_bytes() if path.exists() else download(spec["url"], spec["maximum_bytes"])
        sha = hashlib.sha256(payload).hexdigest()
        if len(payload) > spec["maximum_bytes"] or sha != spec["sha256"]:
            raise ValueError(f"CMU asset identity/size mismatch: {name}")
        if not path.exists():
            path.write_bytes(payload)
        records.append({"name": name, "url": spec["url"], "bytes": len(payload), "sha256": sha})
    provenance = {"subject": "07", "trial": "04", "description": "slow walk", "fps": 120,
                  "retrievedAt": datetime.now(timezone.utc).isoformat(), "files": records,
                  "attribution": ATTRIBUTION, "terms": ["http://mocap.cs.cmu.edu/", "http://mocap.cs.cmu.edu/faqs.php"],
                  "licenseLabel": "CMU stated use terms; not CC-BY", "rawDataCommitted": False}
    provenance_path = destination / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["files"] != records:
            raise ValueError("Existing CMU provenance differs from fetched assets.")
        provenance = previous
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


def load(directory: Path = CACHE):
    provenance = json.loads((directory / "provenance.json").read_text())
    if len(provenance["files"]) != len(ASSETS) or {r["name"] for r in provenance["files"]} != set(ASSETS):
        raise ValueError("Incomplete CMU provenance.")
    for record in provenance["files"]:
        spec = ASSETS[record["name"]]
        sha = hashlib.sha256((directory / record["name"]).read_bytes()).hexdigest()
        if sha != record["sha256"] or sha != spec["sha256"] or record["url"] != spec["url"]:
            raise ValueError("CMU data digest mismatch.")
    skeleton = parse_asf((directory / "07.asf").read_text(encoding="ascii"))
    frames = parse_amc((directory / "07_04.amc").read_text(encoding="ascii"), skeleton)
    return skeleton, frames, provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.cache)
    skeleton, frames, _ = load(args.cache)
    poses = [forward(skeleton, frame)[0] for frame in frames]
    print(json.dumps({"frames": len(frames), "durationS": (len(frames) - 1) / 120, "metresPerUnit": skeleton.scale,
                      "boneCount": len(skeleton.bones), "firstRootM": poses[0]["root"].tolist(),
                      "lastRootM": poses[-1]["root"].tolist(),
                      "boundsM": {key: [np.min([p[key] for p in poses], axis=0).tolist(),
                                      np.max([p[key] for p in poses], axis=0).tolist()]
                                  for key in ("root", "ltibia", "rtibia", "lfoot", "rfoot", "ltoes", "rtoes")}}, indent=2))
