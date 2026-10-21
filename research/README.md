# Optional CPU pretrained-physics checkpoint

This backend runs NVIDIA's pretrained G1 deployment tracker through ONNX Runtime and steps a floating-base Unitree G1 in MuJoCo. It is separate from the procedural browser baseline. The browser physics page replays **recorded simulation transforms**; it does not run the policy live.

**Verified checkpoint:** both 10-second CPU physics trials completed without detected falls or MuJoCo numerical warnings. The saved replay contains actual observations from 1,000 policy calls and 20,000 physics steps, not procedural animation substituted for inference. These are small synthetic-reference smoke tests, not state-of-the-art benchmark results.

## Reproduce on Windows

Use Python 3.11 (the machine's unqualified `python` may select a newer incompatible interpreter). Everything is installed in a project-local virtual environment.

```powershell
py -3.11 -m venv research\.venv
& research\.venv\Scripts\python.exe -m pip install --no-compile --only-binary=:all: -r research\requirements.txt
& research\.venv\Scripts\python.exe research\fetch_assets.py
& research\.venv\Scripts\python.exe -m unittest discover -s research -p "test_*.py" -v
& research\.venv\Scripts\python.exe research\rollout.py --seconds 10
npm run build
npm run preview
```

Open `http://127.0.0.1:4173/physics.html`, also linked from the kinematic playground. The committed JSON replay can be viewed without installing Python or downloading weights; Python is needed only to reproduce inference. Asset setup requires internet access; inference and playback do not. A missing replay is an explicit error, not an invented demonstration. If venv creation was interrupted before pip was installed, run `research\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip` before installing requirements.

The original setup attempt was interrupted by host execution stalls and left partially installed wheels. After the host recovered, `pip install --no-compile --force-reinstall --only-binary=:all: -r research\requirements.txt` repaired the isolated environment and `pip check` passed. No system Python changes or host-safeguard changes were needed.

`--seconds` bounds each of two reference cases to 1-30 seconds. Default output is `public/physics-replay.json`. To keep a new run out of the committed replay, use `--output .research-cache\trial.json`.

## Provenance and licensing

Pinned official source: [NVlabs/ProtoMotions at f69fdcaabff31846b4d2247207417b57dc788828](https://github.com/NVlabs/ProtoMotions/tree/f69fdcaabff31846b4d2247207417b57dc788828).

| Component | Terms and handling |
| --- | --- |
| `g1-bones-deploy/compiled_models/unified_pipeline.onnx` | NVIDIA Open Model License, 24 October 2025; downloaded only to ignored local cache with its license PDF, notice and model card |
| G1 MJCF | Unitree Robotics BSD-3-Clause; attribution/license retained beside the cached description |
| Deployment protocol reference | NVIDIA ProtoMotions Apache-2.0; source license and third-party notice retained in cache |
| Reference motions | Self-authored grounded nominal stand and slow symmetric shoulder/elbow sweep; no BONES or other motion dataset downloaded |
| Replay | Our numerical simulation output, not upstream weights or motion data |

Model SHA-256: `a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c` (22,606,590 bytes).

Applicable source and robot notices are retained in [licenses/](licenses/). Cached model artifacts retain their own NVIDIA license PDF, notice and model card; weights are not committed or served by the browser.

`fetch_assets.py` retrieves a fixed allowlist from the pinned official revision, bounds payload sizes, checks the model against its official Git LFS SHA-256 and records every file digest. The runner verifies these digests before loading. It reads only ONNX, safe YAML, XML and JSON: no `.pt`, `.ckpt`, pickle, PyTorch, upstream Python execution, or custom ONNX plugins. ONNX Runtime telemetry is disabled. Nothing is uploaded.

The full BONES-SEED reference-motion access/license contract was not established for this user. The public presence of a retargeted file is not treated as permission. The optional route therefore **does not reproduce the upstream BONES benchmark**. The small ONNX export avoids the 238 MB training checkpoint and full GPU-oriented framework.

## Contract and deviations

The implementation follows the pinned official [standalone MuJoCo runner](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/deployment/test_tracker_mujoco.py), [input builder](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/deployment/tracker_inputs.py) and compiled YAML metadata.

- 29 joints; exact body, joint and actuator order is asserted against metadata.
- 50 Hz policy, 1,000 Hz physics, 20 simulation substeps per action, single environment and one ONNX inference thread.
- Torso orientation is the anchor input. Pelvis-local angular velocity comes from the free-joint velocity. MuJoCo wxyz rotations are converted to policy xyzw.
- Future references occur at offsets 1, 2, 4 and 8 control steps. Previous **processed joint-position targets**, not raw actor outputs, provide the action-history input.
- The shipped model uses no target-acceleration clamp and EMA alpha 1; incompatible metadata is rejected. Baked stiffness/damping outputs must match configured PD gains.
- The runner retains primitive collisions, inertias, tendons, joint constraints and free-base dynamics. Only zero-density, non-colliding visual mesh geoms and sensors are removed; no STL files are needed. The replay deliberately renders body-coordinate skeletons, not a mesh-accurate G1.
- Passive damping/stiffness/friction loss are zeroed as in the official deployment runner. Armatures and per-joint effort limits come from the frozen safe-YAML config. Unlike the reference runner's stated force-limit TODO, **both the original joint limits and explicit actuator effort limits are retained/enforced** here.
- MuJoCo's default Euler integrator is retained, matching the unmodified MJCF/standalone path, with gravity -9.81 m/s squared.
- The robot starts at the configured nominal joint pose, with foot collision capsules 2 mm above ground and zero velocity. References use that same world frame, so the initial alignment is identity. This runner does not claim support for arbitrary spatially offset reference clips.

## Measurements and interpretation

Each run records the source commit, asset hashes, exact runtime versions, control/physics step counts, numerical warnings, timing, body coordinates, joint tracking error and contact counts.

- **Completion** means all requested finite simulation steps ran; it does not mean tracking succeeded.
- **Fall** means root height below 0.45 m or pelvis tilt above 60 degrees for five consecutive control samples. A detected fall remains flagged even if the robot later recovers. The rollout continues so failure is visible rather than hidden.
- **Joint RMSE** aggregates joint-position error over all 29 joints and sampled control times, including the initial state.
- **Body position error** is the mean Euclidean error across body origins in the common world frame.
- **Contact count** counts MuJoCo contacts, including self-contact; it is not a force measurement or scheduled gait flag.
- **Inference timing** is measured wall time per real ONNX call; total rollout timing includes observation assembly, metric collection and physics. Neither is a controlled hardware benchmark.

These are two deterministic synthetic-reference smoke tests, not a representative motion corpus, robustness study, generalization result, trained policy of our own, physical robot experiment or evidence of state-of-the-art performance.

The test suite checks four deployment contracts and the committed 10-second replay. Replay regression thresholds are joint RMSE below 0.25 rad, mean body-origin error below 0.06 m, root height above 0.7 m, pelvis tilt below 10 degrees, no falls/warnings, and exactly 251 finite 33-body frames per case. These are local smoke-test gates chosen for this checkpoint, not published benchmark criteria. A shorter exploratory run should use `--output` rather than replacing the committed replay.

### Recorded results

The committed `public/physics-replay.json` records the complete provenance and runtime versions: Python 3.11.9, MuJoCo 3.3.7, ONNX Runtime 1.23.2, NumPy 2.2.6, CPUExecutionProvider, one inference thread. All direct and transitive wheel versions from this Windows environment are pinned in `requirements.txt`.

| Metric | Nominal stand | Slow arm sweep |
| --- | ---: | ---: |
| Simulation duration | 10.0 s | 10.0 s |
| Policy calls / physics steps | 500 / 10,000 | 500 / 10,000 |
| Completed / detected falls | Yes / none | Yes / none |
| Joint RMSE | 0.151715 rad | 0.175384 rad |
| Mean body-origin position error | 0.039812 m | 0.037534 m |
| Minimum root height | 0.759022 m | 0.759022 m |
| Maximum pelvis tilt | 4.3352 degrees | 4.6776 degrees |
| Numerical warnings | 0 | 0 |
| Mean / p95 ONNX inference | 53.50 / 130.89 ms | 36.71 / 69.71 ms |
| Wall time for rollout loop | 58.05 s | 40.08 s |
| Simulation / wall-time ratio | 0.172x | 0.250x |

These runs were **slower than real time** on the loaded local machine. No real-time inference claim is made. Earlier two-second setup trials also completed, but are not used instead of the longer recorded results. The replay is deliberately labeled recorded playback and does not imply that a 60 FPS renderer corresponds to 50 Hz live inference.
