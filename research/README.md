# Optional CPU pretrained-physics checkpoint

This backend runs NVIDIA's pretrained G1 deployment tracker through ONNX Runtime and steps a floating-base Unitree G1 in MuJoCo. It is separate from the procedural browser baseline. The browser physics page replays **recorded simulation transforms**; it does not run the policy live.

**Latest checkpoint: [path diagnosis and new causal heading feedback](PATH-TRACKING.md).** One fixed controller reduces root XY RMSE 39.23% and final reference-relative cross-track error 78.49%, preserving two steps per foot and no fall. All three tracking gates still fail; along-track lag and left-foot slip worsen. It changes only supported future torso-yaw inputs, not the physical root or original reference. The common-transform diagnostic's late-joint failure remains recorded alongside its independent one-step resolution. Playback at `/physics.html?replay=path` compares the real baseline and new feedback; no gain search or new dependency.

**Preserved checkpoint: [CMU retarget repair and actual physical steps](CMU-PHYSICS.md).** A finite-difference-verified constrained solver, grounded heel/toe rocking and temporal joint bounds repair the local reference defects. One real 3.72-second trial records two verified steps per foot, +2.499748 m progress and no fall/warnings. Lateral drift (-0.777209 m), root XY RMSE and body tracking still fail the unchanged physical criteria. Original playback remains at `/physics.html?replay=cmu`; it is not reference animation, live inference or a start-from-rest controller.

**Preserved v1: [CMU 07/04 reference rejection](CMU-RETARGET.md).** The first candidate failed contact inference, reach, IK, tendon and velocity checks before physics. Both the completed preflight and its initial reporting-error recovery remain immutable, alongside every later solver version. Raw motion stays in ignored cache.

**Previous checkpoint: [failed-walking contract diagnosis](CONTRACT-AUDIT.md).** Independent source semantics match actual ONNX intermediate observations and physical action scaling. Reference probes change policy output; both unchanged walking trajectories reproduce exactly, with no torque saturation but insufficient support transfer. No runtime contract fix or walking success is claimed. The report includes complete requested/commanded/realized leg evidence, the documentation's tanh discrepancy, and the licensed-reference access boundary.

**Verified checkpoint:** both 10-second CPU physics trials completed without detected falls or MuJoCo numerical warnings. The saved replay contains actual observations from 1,000 policy calls and 20,000 physics steps, not procedural animation substituted for inference. These are small synthetic-reference smoke tests, not state-of-the-art benchmark results.

## Reproduce on Windows

Use Python 3.11 (the machine's unqualified `python` may select a newer incompatible interpreter). Everything is installed in a project-local virtual environment.

```powershell
py -3.11 -m venv research\.venv
& research\.venv\Scripts\python.exe -m pip install --no-compile --only-binary=:all: -r research\requirements.txt
& research\.venv\Scripts\python.exe research\fetch_assets.py
& research\.venv\Scripts\python.exe -m unittest discover -s research -p "test_rollout.py" -v
& research\.venv\Scripts\python.exe research\rollout.py --seconds 10 --output .research-cache\stand-arm-reproduction.json
npm run build
npm run preview
```

Open `http://127.0.0.1:4173/physics.html`, also linked from the kinematic playground. The committed JSON replay can be viewed without installing Python or downloading weights; Python is needed only to reproduce inference. Asset setup requires internet access; inference and playback do not. A missing replay is an explicit error, not an invented demonstration. If venv creation was interrupted before pip was installed, run `research\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip` before installing requirements.

For the complete research test suite, also install the optional pinned tools with `research\.venv\Scripts\python.exe -m pip install --no-compile --only-binary=:all: -r research\diagnostics-requirements.txt -r research\retarget-requirements.txt`, fetch the two selected CMU assets with `research\.venv\Scripts\python.exe research\cmu.py --fetch`, then run `research\.venv\Scripts\python.exe -m unittest discover -s research -p "test_*.py" -v`. See [CMU-PHYSICS.md](CMU-PHYSICS.md#reproduce-and-inspect) for bounded mocap reproduction. The optional retarget tool adds only SciPy 1.16.3; the original standing runner does not need it.

The original setup attempt was interrupted by host execution stalls and left partially installed wheels. After the host recovered, `pip install --no-compile --force-reinstall --only-binary=:all: -r research\requirements.txt` repaired the isolated environment and `pip check` passed. No system Python changes or host-safeguard changes were needed.

`--seconds` bounds each of two reference cases to 1-30 seconds. Default output is `public/physics-replay.json`. To keep a new run out of the committed replay, use `--output .research-cache\trial.json`.

## Provenance and licensing

Pinned official source: [NVlabs/ProtoMotions at f69fdcaabff31846b4d2247207417b57dc788828](https://github.com/NVlabs/ProtoMotions/tree/f69fdcaabff31846b4d2247207417b57dc788828).

| Component | Terms and handling |
| --- | --- |
| `g1-bones-deploy/compiled_models/unified_pipeline.onnx` | NVIDIA Open Model License, 24 October 2025; downloaded only to ignored local cache with its license PDF, notice and model card |
| G1 MJCF | Unitree Robotics BSD-3-Clause; attribution/license retained beside the cached description |
| Deployment protocol reference | NVIDIA ProtoMotions Apache-2.0; source license and third-party notice retained in cache |
| Physical-rollout references | Self-authored grounded nominal stand, slow symmetric shoulder/elbow sweep and bounded synthetic walking; no BONES motion data downloaded |
| CMU reference and physical tracking | Only subject 07 skeleton and trial 04 motion; CMU stated terms, not CC-BY. Raw files stay ignored; v1 was rejected and repaired v2.2 drove one physical trial. [Terms and attribution](CMU-RETARGET.md#source-terms-and-exact-files) |
| Replay | Measured simulation and clearly identified derived reference poses; no weights or raw ASF/AMC files |

Model SHA-256: `a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c` (22,606,590 bytes).

Applicable source and robot notices are retained in [licenses/](licenses/). Cached model artifacts retain their own NVIDIA license PDF, notice and model card; weights are not committed or served by the browser.

`fetch_assets.py` retrieves a fixed allowlist from the pinned official revision, bounds payload sizes, checks the model against its official Git LFS SHA-256 and records every file digest. The runner verifies these digests before loading. It reads only ONNX, safe YAML, XML and JSON: no `.pt`, `.ckpt`, pickle, PyTorch, upstream Python execution, or custom ONNX plugins. ONNX Runtime telemetry is disabled. Inference makes no external requests; cached weights and raw data are not uploaded.

The full BONES-SEED reference-motion access/license contract was not established for this user. The subsequent audit verified [custom eligibility-restricted terms](https://bones.studio/info/seed-license) and gated access; no terms were accepted or eligibility asserted. The public presence of a retargeted file is not treated as permission. The optional route therefore **does not reproduce the upstream BONES benchmark**. The small ONNX export avoids the 238 MB training checkpoint and full GPU-oriented framework.

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
- The original stand/arm/comparison/synthetic-walk paths still start at the configured nominal joint pose, with foot collision capsules 2 mm above ground and zero velocity. The new optional `initial_state` accepts a validated pose/velocity only when it agrees with the custom reference's joints, velocities, full body positions and torso anchor, and satisfies joint/tendon bounds. CMU uses this explicit moving initialization once; no simulated state is reset from the reference afterward.

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

## Fixed matched comparison

`comparison-protocol.json` fixes the protocol **before running the comparison**: two references (stand and arm sweep), three conditions (nominal, seed 17 pushes, seed 29 pushes), and two controllers, for twelve five-second episodes. It does not replace the ten-second replay. Execute:

```powershell
& research\.venv\Scripts\python.exe -m unittest discover -s research -p "test_*.py" -v
& research\.venv\Scripts\python.exe research\evaluate.py
```

The optional `--output` argument selects a separate JSON result; the default is `public/physics-comparison.json`. The result records the exact protocol and its SHA-256, runner source hashes, asset provenance, runtime versions, every episode and pooled summaries. It is saved atomically before execution and after every episode. Interrupted or contract-error runs remain explicitly `running`, not complete.

**Matching:** both controllers use the same model instance with a fresh simulation state per episode, nominal initial pose and grounded root height, physical timestep, gains, effort limits, collision geometry and reference generator. `direct_pd` sends the current reference joint positions straight to the same position actuators; it has no learned correction, feedforward torque, lookahead or ONNX invocation. `onnx` retains its pretrained future-reference/history contract. Both use the same world-frame pelvis forces. Execution order alternates controller-first within reference/condition pairs; timings remain loaded-host measurements, not controlled speed benchmarks.

**Disturbances:** for each non-nominal seed, NumPy PCG64 produces two independent horizontal directions. Each push applies 60 N at the pelvis center of mass, with zero vertical force and zero applied torque, during the half-open intervals 1.0-1.2 s and 3.0-3.2 s: 12 N s per push. Directions are identical across controllers and references for a given seed and are saved numerically. No controller is tuned and no seed is selected based on the results.

**Accounting:** falls use the same definition as the replay and do not stop a rollout. Success means completion without a detected fall, **not** a joint-tracking accuracy pass. Non-finite states/actions, numerical warnings or a 90-second per-episode wall budget terminate that episode, retain its last valid control-sample metrics and record the reason. Contract/setup errors abort rather than being converted into successes. The budget is checked between control steps, so it cannot interrupt an individually stalled native call.

All attempted episodes, including falls and early terminations, contribute to failure rates and sample-weighted observed errors. Joint RMSE is pooled from squared errors, not averaged per-episode RMSE. Full-horizon pooled error is withheld unless every scheduled episode for that controller completed. A partial run shows attempted counts; no missing episode is counted as success. Duration and numerical-warning counts are reported separately. These two references and two perturbation seeds are too small and artificial for a published benchmark, statistical superiority, robustness or generalization claim.

### Measured matched results

Protocol SHA-256: `54701a16747837083a891f160cd59c5f92f8caa62250181e8167f0fb4a3c3379`. The fixed matrix was run once; no perturbations were tuned or trials selected afterward. All twelve episodes completed their five-second horizon: 3,000 controller calls (1,500 real ONNX calls and 1,500 direct reference copies) and 60,000 physics steps, with no early terminations or numerical warnings. The original ten-second replay is unchanged.

| Metric across all six episodes per controller | Pretrained ONNX | Direct PD |
| --- | ---: | ---: |
| Completed without a detected fall | 6 / 6 | 0 / 6 |
| Detected falls / fall rate | 0 / 6 (0%) | 6 / 6 (100%) |
| Early terminations | 0 | 0 |
| Valid simulated duration | 30 s | 30 s |
| Pooled joint RMSE, including fallen states | 0.154867 rad | **0.098948 rad** |
| Mean body-origin world-space error | **0.085157 m** | 0.797560 m |
| Total rollout-loop wall time | 139.43 s | **56.47 s** |

**The pretrained controller did not win every metric.** Direct PD tracked joint angles more closely and ran faster, but it fell even in the nominal cases. Its missing balance feedback is an intentional baseline limitation, not evidence that all classical controllers fail. Small joint error does not imply world-space stability. The learned tracker remained upright here, but the perturbed stand with seed 29 had 0.178432 m mean body-origin error: standing without a fall does not mean staying accurately at the reference position.

Direct PD fell around 1.34-1.38 s, so its second push occurred after falling. Because it also fell without disturbances, this protocol does not isolate a perturbation robustness threshold. It demonstrates the weakness of this unbalanced reference-PD baseline under the declared conditions, not superiority to a classical balance controller.

Every case is retained below. Joint and body errors include the entire five seconds, including time after a fall. Fall times identify the first sample in the five-sample detection streak.

| Reference | Condition | Controller | First fall | Joint RMSE (rad) | Mean body error (m) | Loop wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Stand | Nominal | ONNX | None | 0.150291 | 0.039802 | 20.80 |
| Stand | Nominal | Direct PD | 1.38 s | 0.096642 | 0.784345 | 9.74 |
| Stand | Push seed 17 | ONNX | None | 0.149352 | 0.073722 | 22.76 |
| Stand | Push seed 17 | Direct PD | 1.36 s | 0.099025 | 0.801078 | 9.04 |
| Stand | Push seed 29 | ONNX | None | 0.145390 | 0.178432 | 23.75 |
| Stand | Push seed 29 | Direct PD | 1.34 s | 0.099608 | 0.806015 | 8.69 |
| Arm sweep | Nominal | ONNX | None | 0.170645 | 0.037876 | 21.03 |
| Arm sweep | Nominal | Direct PD | 1.38 s | 0.097854 | 0.782924 | 9.46 |
| Arm sweep | Push seed 17 | ONNX | None | 0.159545 | 0.072285 | 31.87 |
| Arm sweep | Push seed 17 | Direct PD | 1.36 s | 0.100172 | 0.802528 | 11.31 |
| Arm sweep | Push seed 29 | ONNX | None | 0.152631 | 0.108823 | 19.23 |
| Arm sweep | Push seed 29 | Direct PD | 1.34 s | 0.100333 | 0.808472 | 8.23 |

The browser displays this recorded comparison below the independent ten-second replay. A missing or unfinished comparison produces an explicit error without disabling replay; it never substitutes invented results or presents partial wins.

## Bounded physical locomotion experiment

The frozen `locomotion-protocol.json` defines an eight-second slow-walk attempt, with at most two recorded attempts and no overwrite of existing attempt files. The initial second is a nominal stand, followed by four alternating 1.2-second steps and a final hold. Foot placements advance 0.10 m per alternating step, ending at a 0.35 m reference root advance; each swing has 0.04 m clearance. Smooth quintic transfers shift the reference root 0.08 m toward the stance foot. This is a kinematically coherent candidate, **not a dynamically validated walking trajectory**.

The browser playground's two-bone solver cannot represent G1's six-joint leg axes, foot orientation or coupled ankle limits. Instead, the generator uses MuJoCo's exact Jacobians and joint limits, with bounded damped least-squares leg IK, constant foot orientation and stationary world-space stance feet. Joint velocities are finite differences of the resulting coherent poses. Every frame must satisfy position/orientation residual gates, joint and ankle-tendon bounds, and a 4 rad/s reference-joint-velocity ceiling before policy rollout is allowed.

**Contract limitation:** inspection of the pinned ONNX metadata confirms only future joint positions/velocities and torso rotations condition this export. Root position and root linear velocity are not actual ONNX inputs, despite appearing as passthrough metadata. The generated root path is used for reference IK and error measurement only. It is never assigned to simulated root state after initialization. Root progress must emerge from policy torques and contact dynamics.

Before rollout, success is fixed as: all eight seconds completed, no fall or numerical warning, 0.25-0.45 m actual forward root progress, final lateral drift at most 0.15 m, root XY tracking RMSE at most 0.15 m, joint RMSE at most 0.30 rad, mean body-origin error at most 0.15 m, and at least two physically verified steps **per foot**. Support must be present in at least 90% of samples after the initial half-second. Per-foot mean/p95 contact slip must not exceed 0.08/0.20 m/s. All gates must pass; airborne feet or root translation alone cannot count as walking.

At each 50 Hz control sample, foot-floor normal forces are read from MuJoCo contacts; a foot is loaded only above 5 N total normal force. Tangential world velocity at each physical contact point is computed from its Jacobian and actual generalized velocity, then normal-force-weighted within each foot. Mean and p95 slip use only loaded samples; no loaded samples means unavailable, not zero slip. Integrated slip is a sampled estimate, not continuous 1 kHz ground truth. A verified step needs a loaded-to-unloaded-to-loaded transition, at least 0.10 s airborne, 0.02 m rise above liftoff, and 0.05 m forward landing displacement. Sliding while continuously loaded does not count.

```powershell
& research\.venv\Scripts\python.exe -m unittest discover -s research -p "test_locomotion.py" -v
& research\.venv\Scripts\python.exe research\locomotion.py --attempt 1
```

Results go to `public/locomotion/attempt-01.json`. Existing files are never overwritten; use `--output-dir .research-cache\locomotion-reproduction` for independent reproduction. An unsuccessful experiment exits nonzero **after saving** configuration, reference diagnostics, real physics frames, every contact/root sample, failed gates and provenance. A reference-generation failure is recorded explicitly without claiming any physics rollout. The per-attempt wall budget is 120 seconds, checked between control steps. The existing ten-second replay and twelve-case comparison remain separate and unchanged.

### Attempt ledger

1. **Attempt 01 / v1:** eight seconds completed, no fall or numerical warning, but no physically verified steps. Both feet were loaded at all 400 post-initial samples. During intended swings, minimum foot loads were 89.85/84.63 N; actual foot height excursions after the initial hold were only 0.050/0.293 mm. Actual root progress was **-0.025202 m**, not the +0.35 m target. Root XY RMSE 0.253816 m and mean body-origin error 0.216728 m failed their gates despite joint RMSE 0.154050 rad and low contact slip. The robot stood and adjusted its joints; this is **not physical walking**.
2. **Attempt 02 / v2, fixed before its rollout:** the final engineering variant increased foot clearance from 0.04 to 0.08 m and stance-side reference shift from 0.08 to 0.10 m, testing whether stronger leg/lateral motion prompts unloading. Timing, target progress, physical model and every pass criterion were unchanged. Its complete configuration is `locomotion-protocol-02.json`; invoke `research\locomotion.py --attempt 2`. It also completed eight seconds without falling but produced zero verified steps, -0.023673 m root progress, and failed the same four gates. Intended-swing foot loads never fell below 56.51/67.55 N. This was an adaptive engineering attempt, not an independent confirmation or benchmark sample. **Stopped after two attempts, with no physical walking achieved.**

Reference generation passed its numerical gates before the first physical attempt; there were no discarded physical trials or unrecorded parameter sweeps.

| Recorded measure | Attempt 01 | Attempt 02 |
| --- | ---: | ---: |
| Reference clearance / stance-side shift | 0.04 / 0.08 m | 0.08 / 0.10 m |
| Completed duration / numerical warnings | 8 s / 0 | 8 s / 0 |
| Detected falls | 0 | 0 |
| Reference / actual forward progress | +0.35 / **-0.025202 m** | +0.35 / **-0.023673 m** |
| Verified steps, left / right | **0 / 0** | **0 / 0** |
| Root XY RMSE (gate: <= 0.15 m) | **0.253816 m** | **0.254879 m** |
| Mean body-origin error (gate: <= 0.15 m) | **0.216728 m** | **0.219110 m** |
| Joint RMSE (gate: <= 0.30 rad) | 0.154050 rad | 0.158684 rad |
| Loaded samples, left / right | 400 / 400 | 400 / 400 |
| Mean contact slip, left / right | 0.418 / 0.507 mm/s | 0.469 / 0.606 mm/s |
| P95 contact slip, left / right | 0.679 / 0.802 mm/s | 0.781 / 1.519 mm/s |
| Foot height excursion after initial hold, left / right | 0.050 / 0.293 mm | 0.078 / 0.470 mm |
| Loop wall time | 7.977 s | 7.612 s |
| Overall fixed locomotion gates | **Failed** | **Failed** |

Each attempt executed 400 actual ONNX calls and 8,000 physics steps, with 401 force/root samples and 201 recorded body/reference frames. Both reference generators stayed below 0.000001 m foot-position error and satisfied actual G1 joint/tendon constraints; maximum reference joint velocities were 1.570 / 2.770 rad/s. That establishes kinematic feasibility, **not dynamic feasibility or policy compatibility**.

The concrete failure is lack of foot unloading and progress, not a missing GPU or inability to run physics. The controller kept both feet planted while accommodating some joint motion. The export cannot directly consume the planned root position or linear velocity, but these experiments do not isolate that omission as the cause. A compatible, dynamically credible locomotion reference and/or a controller with the needed tracking contract must be validated before claiming walking; repeatedly increasing motion amplitudes without such evidence is not justified.

Timing improved when this host was less loaded: near-real-time loop timing in these two runs is not a real-time guarantee, a controlled comparison with earlier runs, or evidence of successful locomotion.

`public/locomotion/attempt-01.json` and `attempt-02.json` retain complete failed outputs. Their embedded protocols and execution-time source hashes preserve the attempted configurations. The small browser summary is derived without changing either artifact:

```powershell
& research\.venv\Scripts\python.exe research\locomotion.py --summarize
```

The physics page's locomotion section reports **no physical walking achieved**, links both full artifacts, and does not substitute the kinematic playground or standing replay. Tests independently recompute every stored locomotion metric/gate from recorded samples and check that attempts cannot be overwritten or extended beyond the declared bound.
