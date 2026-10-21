# Kinetic / Humanoid Lab

A local, interactive 3D humanoid with analytic limb inverse kinematics, a procedural gait generator, reachable hand-target interaction, and live numerical diagnostics. Built with TypeScript, Three.js and Vite; all robot geometry is authored in code. No external models, API keys, paid services, or policy weights.

**This is a kinematic baseline, not a learned physics controller or a state-of-the-art result.** Feet follow prescribed paths; the body does not simulate forces, contact dynamics, balance, or falls. Small IK residuals are implementation checks, not evidence of physical tracking performance.

![Kinetic desktop demo, captured by the software-rendered browser smoke test](docs/demo.png)

## Run locally

Requires Node.js 22.18+ (tested on Node 24.22.1), npm, and a WebGL2-capable browser.

```powershell
npm ci
npm run dev
```

Open the loopback URL printed by Vite. For the production demo:

```powershell
npm run build
npm run preview
```

Open **http://127.0.0.1:4173**. The server binds to loopback, not the public network. Stop it with Ctrl+C. After dependency installation the app makes no external asset, font, model, analytics, or API requests.

## Explore

| Control | Behavior |
| --- | --- |
| Idle / Walk / Jog / March | Resting motion, translating gaits, or high stepping in place |
| Tempo | Scales cadence and translation together; zero stops the gait |
| Steering / A and D / arrow keys | Turns the body; Center clears the steering slider |
| Amber hand target | Drag in a camera-facing plane; X/Y/Z sliders are a keyboard-accessible alternative |
| IK toggle | Switches the right arm between target reaching and procedural arm swing |
| Orbit / zoom / camera presets | Drag empty scene space, scroll, or choose front / three-quarter views |
| Joint overlay | Draws the actual two-bone chains over the robot |
| Pause / Space | Freezes simulation time and locomotion; camera and target editing remain available |
| Reset | Returns time, root position, heading and hand target to their initial values; preserves selected mode, tempo and paused state |
| Export snapshot | Downloads current mode, target, residuals, contacts, render statistics and position as JSON |

All coordinates and IK residuals are in meters internally. The UI displays errors in millimeters. The target is body-relative: it travels and turns with the humanoid. Out-of-workspace targets clamp to the closest point on the arm's reach shell, with an explicit residual rather than a false success.

## Model and measurable checks

`src/kinematics.ts` is independent of rendering. It contains a two-bone geometric IK solver with degenerate-direction handling, four motion presets and deterministic time integration. Leg segments are each 0.48 m. Arm segments are 0.30 m and 0.29 m. Stance takes 62% of a gait cycle; stance travel cancels root translation during straight walking, while smooth swing paths clear the flat ground. The prescribed hip height stays within leg reach.

```powershell
npm test
npm run build
npx playwright install chromium --only-shell
npm run test:browser
```

The numerical tests print their measured maxima and enforce these thresholds:

| Check | Coverage | Required result |
| --- | --- | --- |
| IK segment-length preservation | 10,000 deterministic targets, including unreachable targets | Absolute error < 0.0000001 m |
| Reachable target residual | Reachable subset of the same targets | < 0.0000001 m |
| Stance endpoint height | 19,200 poses across 4 modes and 4 tempo values | < 0.1 mm |
| Straight-line stance foot slide | 7,200 simulation steps across walk/jog and 3 tempos | < 0.1 mm per step |
| Degenerate targets | Coincident root, parallel pole, far target, invalid inputs | Finite valid solutions or explicit input error |
| Gait boundary continuity | Stance/swing and cycle boundaries | Position difference < 0.0000001 m |
| Pause, reset, replay | Identical input sequences | Exact deterministic state equality |

Browser tests run serially to limit memory use. They use Chromium's software WebGL backend so the smoke test does not depend on a dedicated GPU. They exercise all motion modes, pause/reset, tempo, steering, target reach limits, direct target dragging, keyboard pause, overlay, export and mobile layout. Screenshots are written to `test-results/kinetic-desktop.png` and `test-results/kinetic-mobile.png`. The exported FPS is measured locally, not a promised hardware-independent rate.

### Recorded local run (21 October 2026)

Node 24.22.1 on Windows: **6 numerical tests and 3 browser tests passed**, and the TypeScript / Vite production build passed. The browser suite took about 2.4 minutes with one worker and software rendering; trace recording is disabled to avoid memory-heavy screenshots during every action.

| Measured diagnostic | Maximum |
| --- | --- |
| IK segment-length error, 10,000 targets | 1.05e-15 m |
| Reachable IK target residual | 2.29e-16 m |
| Stance endpoint height error, 19,200 poses | 1.39e-16 m |
| Straight-line stance slide per step, 7,200 steps | 2.07e-15 m |

These near-machine-precision results are expected for an analytic geometric construction. They do **not** measure physical stability, contact forces, learned-motion quality, or benchmark success rate. The live scene reported 7,938 rendered triangles and 43 draw calls in one sampled pose. Software-rendered screenshot runs were substantially slower than the embedded browser, which sampled approximately 60 FPS; neither is a controlled performance benchmark. Vite emits a non-fatal bundle-size advisory for the approximately 525 kB minified application including Three.js (134 kB gzip).

## What this does not establish

- There is no physics engine, torque or PD controller, collision avoidance, self-collision test, terrain adaptation, learned motion prior, motion capture dataset, text-to-motion model, or real-robot transfer.
- Contact indicators represent the **scheduled kinematic stance**, not force sensors or simulated contacts.
- Straight-line stance cancellation does not extend to turning; turning can produce visible foot sliding. Mode/tempo changes are immediate and not transition-blended. Swing trajectories are position-continuous but not velocity-matched at the stance boundary.
- The displayed root travel is displacement from the origin, not accumulated path length. The floor and camera follow the root; the visible ring is a moving reference, not a fixed platform.
- The 100 ms elapsed-time cap intentionally slows simulation after long stalls rather than jumping forward. Determinism requires the same time-step/input sequence.
- The robot has fixed proportions and no explicit anatomical joint limits. The reach solver prevents segment stretching but not body intersections.
- Integrated Intel HD Graphics 520 is a target for lightweight browser use, not a training platform. Actual browser compatibility and render rate depend on drivers, memory pressure and WebGL2 support.

## Research context: primary sources, not reproduced results

The following references contextualize the distance between this baseline and modern learned humanoid control. Their results are **author-reported**, not measured by this repository, and are not directly comparable to our geometric residual tests.

| Work | Relevant idea | Distinction from this demo |
| --- | --- | --- |
| [SuperMimic (2026)](https://arxiv.org/abs/2610.28480), [project](https://supermimic.github.io) | Differentiable world model plus motion tracker; authors report normalized return above 0.90 after 3 hours on one H100, 25 failures among 129,514 training-set benchmark motions, and zero after 9.8 hours on two GPUs | Training-set coverage is not generalization. No weights, training, physical tracking or benchmark reproduction here |
| [SWAP (2026)](https://arxiv.org/abs/2610.25711) | High-level diffusion collision avoidance with low-level RL tracking; authors report an average 34% improvement in targeted rewards | No collision reasoning or RL in this baseline; no reproduction |
| [MaskedMimic (2024)](https://arxiv.org/abs/2409.14393) | Unified masked-motion-inpainting physics-based control | A foundational learned-control comparison, not a capability this procedural model implements |

SuperMimic's [browser-demo license](https://supermimic.github.io/demo/LICENSE.html) is proprietary and permits personal browser evaluation only, with restrictions on redistribution, derivatives, automated access and evaluation of other systems using its outputs. **No code, weights, assets or outputs from that demo are included, downloaded or automated here.**

### A concrete next research step

[NVIDIA ProtoMotions installation documentation](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/installation.rst) and [pretrained-model documentation](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/pretrained_models.rst) describe a CPU-only MuJoCo route (Python 3.10+, one environment) for the G1 deployment tracker. The `g1-bones-deploy` tracker and `g1_bones_seed_mini` motions could support actual physical tracking measurements on modest hardware. This is a **documented future path, not installed or tested here**.

That tracker consumes reference motion, not text; its 29-action mapping, PD gains, observation history and joint ordering must match the deployment contract. The source is Apache-2.0; weights use the NVIDIA Open Model License, with separate robot and motion-data terms. SMPL MaskedMimic checkpoints are IsaacLab-only and should not be confused with this G1 deployment route. Before claiming progress toward published results, reproduce a named physical benchmark with a pinned model/data version, episode count, tracking errors, failure definition, runtime, and hardware.

## Structure

```text
src/kinematics.ts        Renderer-independent analytic IK and gait
src/main.ts              Geometry, scene, interaction and diagnostics
src/style.css            Responsive laboratory interface
tests/kinematics.test.ts Numerical invariants and motion checks
tests/browser.spec.ts    Browser controls, reach interaction and screenshots
```

Three.js is MIT-licensed; no third-party humanoid assets are bundled. Dependency versions are locked in `package-lock.json`.
