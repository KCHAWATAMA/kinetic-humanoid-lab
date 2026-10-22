# Failed walking: deployment-contract diagnosis

**No implementation mismatch was found in the inspected deployment contract. Walking is still not achieved.** Reference changes reach the real ONNX observation and alter its PD output, but those commands do not reproduce the requested leg posture or transfer enough support to unload either foot. Torque clipping is not the explanation in these runs. This isolates the failure beyond input assembly; it does **not** establish whether the cause is an unsuitable reference, learned-policy behavior, or remaining sim-to-sim differences.

Evidence: [`public/diagnostics/contract-audit-v1.json`](../public/diagnostics/contract-audit-v1.json), available from the preview at `http://127.0.0.1:4173/diagnostics/contract-audit-v1.json`. The 0.89 MB record contains protocol/source/asset hashes, graph checks, all probes, per-joint measurements, and 50 Hz requested/commanded/realized leg traces. It is not a new walking replay or an improved benchmark result.

## Fixed scope and preservation

The protocol was written before measurement: one eight-second stand and diagnostic re-executions of the **same** eight-second walking protocols 01/02, followed by no parameter search. At the stand's 1.60 s state, hold all robot observations and processed-action history fixed and change one future knee/time slot, one velocity channel, one reference rotation, or the complete historical swing reference. This is a causal input-sensitivity test, not a physically coherent new walking attempt.

The three full runs executed 1,200 policy calls and 24,000 physics steps, with zero falls/warnings. Wall times, including instrumentation, were 10.04 / 9.85 / 9.73 s. Host load and added diagnostics preclude comparison with earlier timing measurements.

Both historical walking root trajectories reproduced **exactly** at all 401 samples; joint RMSE, body error, minimum height and maximum tilt also had zero numerical difference. All five historical JSON artifacts retain their SHA-256 values. Attempts 01/02 and the matched comparison were not overwritten. No correction was justified, so the twelve-case comparison was not needlessly rerun. The only shared-runner change is an optional read-only substep observer, independently checked not to change trajectories.

## Contract, checked against source and actual graph

All upstream source/metadata references below are pinned to [`f69fdcaabff31846b4d2247207417b57dc788828`](https://github.com/NVlabs/ProtoMotions/tree/f69fdcaabff31846b4d2247207417b57dc788828). Source text is inspected and hashed, never imported or executed. The original ONNX SHA-256 remains `a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c`.

| Surface | Authoritative semantics and observed match |
| --- | --- |
| Inputs | Eight float32 inputs: `current_anchor_rot [B,4]`, `current_dof_pos [B,29]`, `current_dof_vel [B,29]`, `current_root_local_ang_vel [B,3]`, `historical_processed_actions [B,1,29]`, `mimic_future_anchor_rot [B,4,4]`, `mimic_future_dof_pos [B,4,29]`, `mimic_future_dof_vel [B,4,29]`. Actual graph, YAML and runtime agree; B=1 here. |
| Graph-consumed versus passthrough | All eight inputs are structural ancestors of the actor output. Numeric probes demonstrate value sensitivity. Current anchor position and future anchor position/linear/angular velocity are metadata passthrough, not graph inputs. No root displacement, root speed, terrain height or contact-force input is secretly required. |
| Joint/actuator order and axes | 29 joints in the YAML/MJCF order; every actuator maps to its corresponding joint. Independent XML checks verify hinge axes. Every actuator has unit scalar gear. Exact arrays are in the artifact. |
| Quaternion and velocity frames | MuJoCo wxyz is converted to xyzw. Torso `torso_link` is anchor body index 16 excluding world; pelvis is body 0. Free-joint `qvel[3:6]` is already pelvis-local angular velocity; joint velocities are `qvel[6:]`. A nonidentity, articulated fixture also matches rotation of world `cvel` back into the pelvis frame. |
| Physical input units | Joint angles are raw radians, velocities rad/s, quaternions unit xyzw. No default-pose subtraction or user-side normalization. |
| Observation preprocessing | Independent rotation-matrix implementation of upstream `compute_humanoid_reduced_coords_observations`, `build_reduced_coords_target_poses` and `quat_to_tan_norm` matches the actual graph's 349-feature tensor. Layout: 64 proprioceptive features, four 64-feature targets, 29 previous processed targets. Each future target has relative rotation x/z basis vectors, velocities, then positions. |
| Normalization | Baked mean and denominator, then clipping to [-5,5]. Tested against internal graph outputs, with no change to ordinary inference. Maximum raw-feature disagreement across the held-state probes was 1.20e-7. None of these eleven probes clipped a normalized feature; this is not a claim about every rollout state. |
| Action units, offsets and scales | `joint_pos_targets = pd_action_offset + action_scale * actions`, in radians. Initializers match frozen config; offset equals nominal pose. Our runner uses that output directly, without applying scale twice or treating raw actor values as joint angles. |
| Documentation discrepancy | The general deployment prose calls actions tanh-bounded. **This exact graph has no Tanh node or action Clip**: it exposes the deterministic actor mean, then affine PD scaling. The existing runner already matches the graph, so adding tanh would be a new unsupported behavior change. |
| PD and limits | Metadata stiffness/damping agree with config and baked outputs. Force equals clipped `kp*(target-actuator_length)-kd*actuator_velocity`; maximum measured equation residual was 1.07e-14 N m. Config armature and effort limits are applied. Our force-limit enforcement is an intentional difference from the standalone runner's TODO, consistent with the framework simulator; no limit was reached here. |
| History, initialization and filtering | Initial processed history is 29 zeros, then the last applied physical PD target, matching the standalone runner/input builder. One history frame, not the training buffer's unused 32-state history. Runtime checks found zero history discrepancy. EMA alpha=1 and acceleration clamp=null: neither filter applies. |
| Timing and future sampling | 0.02 s control; offsets [1,2,4,8] mean +0.02/+0.04/+0.08/+0.16 s, not sequential frames or zero-based offsets. Runtime comparison against generated frames found zero discrepancy. 0.001 s physics with 20 substeps follows deployment metadata; training config's 200 Hz physics also has 50 Hz control but is not the deployment substep rate. |
| Ground/root and reference alignment | Flat z=0 plane, Euler integration, gravity -9.81, foot/floor sliding friction 1. The same G1 collision primitives, inertias, free base and ankle tendons are retained. Initial reference and robot pose coincide; heading alignment is identity. Root initialized once with feet 2 mm above ground, zero velocities. Reference root translation drives IK/metrics, never teleports the simulated root. Arbitrarily offset external clips are not supported by this limited runner. |

Primary implementations: [standalone runner](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/deployment/test_tracker_mujoco.py), [input assembly](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/deployment/tracker_inputs.py), [future sampling](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/deployment/motion_utils.py), [proprioception](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/protomotions/envs/obs/humanoid.py), [target observations](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/protomotions/envs/obs/target_poses.py), [framework MuJoCo PD](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/protomotions/simulator/mujoco/simulator.py).

## Controlled probes: the reference is not being ignored

All observations/history are held at the same physical standing state at 1.60 s. Reported deltas are the maximum absolute change across **all 29 output targets**, not necessarily the changed knee.

| Reference-only change from stand | Changed input elements | Maximum PD target change |
| --- | ---: | ---: |
| Left knee +0.1 rad at offset 1 | 1 | 0.004447 rad |
| Left knee +0.1 rad at offset 2 | 1 | 0.003500 rad |
| Left knee +0.1 rad at offset 4 | 1 | 0.005455 rad |
| Left knee +0.1 rad at offset 8 | 1 | 0.004340 rad |
| Right knee +0.1 rad at offset 8 | 1 | 0.006834 rad |
| Left knee velocity +0.2 rad/s at offset 8 | 1 | 0.002019 rad |
| First future torso yaw +5 degrees | 2 quaternion components | 0.025062 rad |
| Complete attempt-01 swing targets | 96 | 0.073839 rad |
| Complete attempt-02 swing targets | 96 | 0.097124 rad |
| Passthrough reference body position only | 0 graph inputs | Exactly 0 |

No observation-normalization clipping occurred at these probes. The unchanged-position probe confirms the documented observability boundary, but does **not** prove that omitted root channels caused failed walking: a reduced-coordinate tracker can in principle walk using appropriate joint/orientation targets.

## Requested, commanded, realized, and supported

At the first swing midpoint, **t=1.60 s**, the left knee angles were:

| Case | Requested reference | Newly commanded PD target | Realized joint | Actual left/right foot normal force |
| --- | ---: | ---: | ---: | ---: |
| Stand | 0.669000 rad | 0.240468 rad | 0.277688 rad | 166.25 / 160.72 N |
| Attempt 01 | 0.938850 rad | 0.360182 rad | 0.372961 rad | 107.14 / 221.64 N |
| Attempt 02 | 1.165183 rad | 0.444810 rad | 0.452706 rad | 84.56 / 245.87 N |

The command is issued at the sample time and governs the next 20 substeps; the realized state at that time reflects earlier commands. PD targets are torque-producing setpoints, not guaranteed realized reference poses. The reference/command gap is already large at the policy output; it is not just a renderer discrepancy.

Over the entire post-hold interval, requested / commanded / realized **left-knee range** was 0.3533 / 0.2365 / 0.1986 rad in attempt 01, and 0.6136 / 0.3756 / 0.3014 rad in attempt 02. Thus legs do move; "both feet remain planted" does not mean zero joint motion.

Across **all 24,000 substeps and all 29 actuators**, zero force-limit saturations occurred. Peak utilization was 42.92% of the applicable limit. Increasing effort limits therefore has no evidence-based justification here.

Support transfer was partial: minimum post-hold left/right loads were 89.85/84.63 N in attempt 01 and 56.51/67.55 N in attempt 02. Minimum load shares were 27.22%/25.69% and 16.90%/20.29%. Neither foot approached the 5 N unloading threshold. Foot height excursions remained 0.050/0.293 mm and 0.078/0.470 mm. Historical zero-step counts and negative progress are unchanged.

**Known:** input wiring, source observation semantics, graph scaling, history/lookahead, physical actuator mapping and finite effort limits behave as specified in these checks. The policy responds to references but supplies insufficient motion/support transfer to meet these walking criteria.

**Unknown:** whether a known-compatible, dynamically credible walking clip succeeds with this exact stack. Kinematic IK residuals and joint limits do not establish dynamic feasibility or distribution compatibility. No positive locomotion control has been run. This diagnosis cannot prove universal policy incapability, diagnose its learned internals, or rule out every IsaacLab-to-MuJoCo difference.

## Smallest reference path and concrete access boundary

The most direct compatibility control is **one short, authorized G1 walking clip from the tracker's documented source distribution**, converted to safe numeric arrays, not a new model or a large training install. The official `g1_bones_seed_mini.pt` LFS pointer declares 24,555,451 bytes and SHA-256 `fa36ece31f99c56f218e51d0db54def318fae163f05c4b66eb7b41649336ef5e`; only the pointer was inspected, not the data.

**It is not an unrestricted download path.** On 22 October 2026, the [public dataset API](https://huggingface.co/api/datasets/bones-studio/seed) reported gated access and custom `bones-seed-license` terms. The [primary license](https://bones.studio/info/seed-license) restricts eligibility to qualifying academic institutions or qualifying startups, prohibits raw-data redistribution, and requires agreement/authority to bind the entity. The user's eligibility and authorization are not established. A public GitHub sample is not treated as a way around those restrictions. The official loader also uses `torch.load(..., weights_only=False)`; it is not adopted here. A rights-cleared, non-pickle clip/export is the smallest practical input needed for that positive control. No dataset was downloaded, no eligibility was attested, and no terms were accepted on the user's behalf.

An independently licensed **source** alternative is one raw CMU ASF/AMC clip, for example [subject 07, trial 04, slow walk at 120 Hz](http://mocap.cs.cmu.edu/search.php?subjectnumber=07), plus its skeleton. [CMU's own terms](http://mocap.cs.cmu.edu/) permit research/commercial-product use, prohibit directly reselling the data, request acknowledgment, and prohibit database crawling. These are CMU's stated terms, **not an asserted CC-BY license**. Only the targeted listing/terms were read. This avoids SMPL/AMASS and BONES eligibility, but requires G1 retargeting, correct units/axes and 50 Hz interpolation, plus independent dynamic validation. It is **not** a verified drop-in reference and has not been downloaded or tested. No currently authorized, ready-to-run G1 positive-control clip is claimed.

## Reproduce without touching historical evidence

Use the existing Python 3.11 environment. Optional graph inspection adds ONNX 1.19.1 and ml_dtypes 0.5.4 only (about 16.7 MB wheel download); runtime dependencies remain unchanged.

```powershell
& research\.venv\Scripts\python.exe -m pip install --no-compile --only-binary=:all: -r research\diagnostics-requirements.txt
& research\.venv\Scripts\python.exe research\diagnose.py --fetch-sources --output .research-cache\contract-audit-reproduction.json
& research\.venv\Scripts\python.exe -m unittest discover -s research -p "test_*.py" -v
```

The diagnostic refuses an existing output path. `--fetch-sources` retrieves the pinned allowlisted source text into ignored cache; omit it for offline reproduction once cached. No model/data download beyond the existing verified ONNX setup is needed. The graph's intermediate tensors are exposed only in an in-memory diagnostic session; neither original weights nor the rollout inference session is modified. Internal-output checks compare instrumented and ordinary outputs.

Each case has a 120 s wall budget checked between control steps. Failures retain explicit failed status and available evidence, never a success-shaped fallback. These bounds do not interrupt a stalled native call. Only completed runs with historical trajectories preserved receive complete status. Software checks protect measurement semantics; they do not turn the failed physical walking criteria into success.

Validation for this diagnosis: all 26 Python checks passed, including five new graph/telemetry/evidence checks; all six kinematic tests and the TypeScript/Vite build passed. The existing end-to-end physics replay test passed in two minutes, exercising replay controls and the preserved comparison/failed-locomotion panels. The new artifact returned HTTP 200 with complete status and all three cases. The other seven unchanged browser checks were not rerun for this diagnostics-only change.
