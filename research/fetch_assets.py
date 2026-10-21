"""Fetch only the pinned official G1 ONNX deployment artifacts, never motion data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

REVISION = "f69fdcaabff31846b4d2247207417b57dc788828"
REPOSITORY = "https://github.com/NVlabs/ProtoMotions"
RAW = f"https://raw.githubusercontent.com/NVlabs/ProtoMotions/{REVISION}/"
MEDIA = f"https://media.githubusercontent.com/media/NVlabs/ProtoMotions/{REVISION}/"
MODEL_DIR = "data/pretrained_models/motion_tracker/g1-bones-deploy"
MODEL_SHA256 = "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"
MODEL_BYTES = 22606590
FILES = [
    f"{MODEL_DIR}/compiled_models/unified_pipeline.onnx",
    f"{MODEL_DIR}/compiled_models/unified_pipeline.yaml",
    f"{MODEL_DIR}/resolved_configs_inference.yaml",
    f"{MODEL_DIR}/LICENSE.pdf",
    f"{MODEL_DIR}/NOTICE.txt",
    f"{MODEL_DIR}/MODEL_CARD.md",
    "protomotions/data/assets/mjcf/g1_holo_compat.xml",
    "protomotions/data/assets/mesh/G1/LICENSE",
    "LICENSE.md",
    "legal/THIRD-PARTY SOFTWARE NOTICES AND ASSET LICENSES - ProtoMotions.txt",
]


def download(url: str, maximum_bytes: int) -> bytes:
    with urlopen(url.replace(" ", "%20"), timeout=90) as response:
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"Download exceeds declared bound: {url}")
    return payload


def fetch(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in FILES:
        payload = download(RAW + relative, 200_000)
        expected_sha = None
        if payload.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            pointer = payload.decode("ascii")
            sha = re.search(r"^oid sha256:([a-f0-9]{64})$", pointer, re.MULTILINE)
            size = re.search(r"^size ([0-9]+)$", pointer, re.MULTILINE)
            if sha is None or size is None:
                raise ValueError(f"Invalid LFS pointer: {relative}")
            expected_sha, expected_bytes = sha[1], int(size[1])
            if expected_sha != MODEL_SHA256 or expected_bytes != MODEL_BYTES:
                raise ValueError(f"Unexpected model identity: {relative}")
            payload = download(MEDIA + relative, expected_bytes)
            if len(payload) != expected_bytes:
                raise ValueError(f"Truncated model: {relative}")
        actual_sha = hashlib.sha256(payload).hexdigest()
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"Model SHA-256 mismatch: {relative}")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append({"path": relative, "bytes": len(payload), "sha256": actual_sha})
        print(f"{len(payload):>10} bytes  {relative}", flush=True)
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "model_license": "NVIDIA Open Model License, 2025-10-24",
        "robot_license": "BSD-3-Clause, Unitree Robotics",
        "source_license": "Apache-2.0",
        "motion_data_downloaded": False,
        "pickle_artifacts_downloaded": False,
        "files": records,
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(".research-cache/protomotions"))
    args = parser.parse_args()
    fetch(args.destination)
