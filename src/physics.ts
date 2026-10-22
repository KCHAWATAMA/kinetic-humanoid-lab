import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadLocomotion } from "./locomotion";
import { loadMocap } from "./mocap";
import "./style.css";
import "./physics.css";

type Point = [number, number, number];
type Frame = {
  time: number; body: Point[]; reference: Point[]; jointRmsRad: number;
  rootHeightM: number; contacts: number; fallen: boolean;
};
type Trial = { name: string; metrics: Record<string, number | boolean | null>; frames: Frame[] };
type Replay = {
  schemaVersion: number; kind: string; policy: string; reference: string; createdAt: string;
  bodyNames: string[]; parents: number[]; cases: Trial[];
  provenance: { revision: string };
  locomotion?: {
    passed: boolean; gates: Record<string, boolean>;
    metrics: { verified_steps_per_foot: number[]; forward_progress_m: number; final_lateral_drift_m: number;
      root_xy_rmse_m: number; mean_contact_slip_m_s: number[]; p95_contact_slip_m_s: number[] };
  };
};
type ComparisonEpisode = {
  name: string; condition: string; controller: "onnx" | "direct_pd"; termination: string | null;
  metrics: {
    completed: boolean; fell: boolean; simulated_seconds: number; first_fall_time_s: number | null;
    joint_rmse_rad: number; mean_body_position_error_m: number; wall_seconds: number;
  };
};
type ComparisonSummary = {
  controller: "onnx" | "direct_pd"; attempted_episodes: number; completed_episodes: number;
  successes: number; falls: number; early_terminations: number; observed_joint_rmse_rad: number;
  observed_mean_body_error_m: number; simulated_seconds: number; wall_seconds: number;
};
type Comparison = { protocolSha256: string; episodes: ComparisonEpisode[]; summary: ComparisonSummary[] };
const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const mocapReplay = new URLSearchParams(location.search).get("replay") === "cmu";
const replayUrl = mocapReplay ? "./mocap-v2/physics-replay.json" : "./physics-replay.json";
function assertReplay(value: unknown): asserts value is Replay {
  if (!record(value) || value.schemaVersion !== 1 || value.kind !== "recorded-pretrained-policy-physics"
    || typeof value.policy !== "string" || typeof value.reference !== "string" || typeof value.createdAt !== "string"
    || !record(value.provenance) || typeof value.provenance.revision !== "string"
    || !Array.isArray(value.bodyNames) || value.bodyNames.length !== 33 || !value.bodyNames.every(n => typeof n === "string")
    || !Array.isArray(value.parents) || value.parents.length !== 33
    || !value.parents.every((n, i) => Number.isInteger(n) && (i === 0 ? n === -1 : n >= 0 && n < i))
    || !Array.isArray(value.cases) || value.cases.length === 0) throw new Error("Unsupported physics replay schema");
  for (const trial of value.cases) {
    if (!record(trial) || typeof trial.name !== "string" || !record(trial.metrics)
      || !Array.isArray(trial.frames) || trial.frames.length < 2) throw new Error("Invalid rollout metrics or frames");
    const metrics = trial.metrics;
    if (!["simulated_seconds", "joint_rmse_rad", "mean_inference_ms", "min_root_height_m"].every(key => finite(metrics[key]))
      || typeof metrics.fell !== "boolean" || typeof metrics.completed !== "boolean"
      || !Object.values(metrics).every(item => item === null || typeof item === "boolean" || finite(item))) throw new Error("Invalid rollout metrics");
    let previousTime = -1;
    for (const frame of trial.frames) {
      if (!record(frame) || !finite(frame.time) || frame.time <= previousTime || !finite(frame.jointRmsRad)
        || !finite(frame.rootHeightM) || !finite(frame.contacts) || typeof frame.fallen !== "boolean"
        || ![frame.body, frame.reference].every(points => Array.isArray(points) && points.length === 33
          && points.every(p => Array.isArray(p) && p.length === 3 && p.every(finite)))) throw new Error("Invalid sampled physics state");
      previousTime = frame.time;
    }
    if (mocapReplay) {
      if (value.experimentId !== "g1-cmu-contact-v2.2" || !record(value.locomotion)
        || typeof value.locomotion.passed !== "boolean" || !record(value.locomotion.gates)
        || Object.keys(value.locomotion.gates).length !== 12
        || !Object.values(value.locomotion.gates).every(g => typeof g === "boolean")
        || value.locomotion.passed !== Object.values(value.locomotion.gates).every(g => g === true)
        || !record(value.locomotion.metrics)) throw new Error("Incomplete measured mocap locomotion outcome");
      const m = value.locomotion.metrics;
      if (!["forward_progress_m", "final_lateral_drift_m", "root_xy_rmse_m"].every(key => finite(m[key]))
        || !Array.isArray(m.verified_steps_per_foot) || m.verified_steps_per_foot.length !== 2
        || !m.verified_steps_per_foot.every(v => Number.isInteger(v) && v >= 0)
        || !["mean_contact_slip_m_s", "p95_contact_slip_m_s"].every(key => {
          const pair = m[key];
          return Array.isArray(pair) && pair.length === 2 && pair.every(v => finite(v) && v >= 0);
        })) throw new Error("Invalid measured mocap locomotion metrics");
    }
  }
}

function assertComparison(value: unknown): asserts value is Comparison {
  if (!record(value) || value.schemaVersion !== 1 || value.kind !== "matched-controller-comparison" || value.status !== "complete"
    || typeof value.protocolSha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.protocolSha256)
    || !record(value.protocol) || value.protocol.id !== "g1-synthetic-matched-v1"
    || !Array.isArray(value.episodes) || value.episodes.length !== 12
    || !Array.isArray(value.summary) || value.summary.length !== 2) throw new Error("Comparison is incomplete or has an unsupported schema");
  const keys = new Set<string>();
  for (const episode of value.episodes) {
    if (!record(episode) || !["stand", "arm_sweep"].includes(String(episode.name))
      || !["nominal", "push_17", "push_29"].includes(String(episode.condition))
      || !["onnx", "direct_pd"].includes(String(episode.controller))
      || !(episode.termination === null || typeof episode.termination === "string")
      || !record(episode.metrics)) throw new Error("Invalid comparison episode");
    const metrics = episode.metrics;
    if (typeof metrics.completed !== "boolean" || typeof metrics.fell !== "boolean"
      || !(metrics.first_fall_time_s === null || finite(metrics.first_fall_time_s))
      || metrics.fell !== (metrics.first_fall_time_s !== null)
      || !["simulated_seconds", "joint_rmse_rad", "mean_body_position_error_m", "wall_seconds"].every(key => finite(metrics[key]) && metrics[key] >= 0)
      || !finite(metrics.simulated_seconds) || metrics.simulated_seconds > 5
      || (metrics.completed && metrics.simulated_seconds !== 5)) {
      throw new Error("Invalid comparison measurements");
    }
    keys.add(`${episode.name}/${episode.condition}/${episode.controller}`);
  }
  if (keys.size !== 12) throw new Error("Comparison contains missing or duplicate matched cases");
  for (const [index, summary] of value.summary.entries()) {
    if (!record(summary) || summary.controller !== ["onnx", "direct_pd"][index]
      || !["attempted_episodes", "completed_episodes", "successes", "falls", "early_terminations",
        "observed_joint_rmse_rad", "observed_mean_body_error_m", "simulated_seconds", "wall_seconds"].every(key => finite(summary[key]))
      || summary.attempted_episodes !== 6) throw new Error("Invalid comparison summary");
    const episodes = value.episodes.filter(episode => episode.controller === summary.controller);
    const completed = episodes.filter(episode => episode.metrics.completed).length;
    if (summary.completed_episodes !== completed || summary.early_terminations !== 6 - completed
      || summary.falls !== episodes.filter(episode => episode.metrics.fell).length
      || summary.successes !== episodes.filter(episode => episode.metrics.completed && !episode.metrics.fell).length) {
      throw new Error("Comparison summary disagrees with its episodes");
    }
  }
}

const app = document.querySelector<HTMLDivElement>("#physics-app")!;
app.innerHTML = `
  <header class="topbar"><a class="brand" href="./"><span class="brand-mark">K</span>KINETIC <span class="brand-divider">/</span><span class="brand-sub">PHYSICS CHECKPOINT</span></a><a class="back-link" href="./">Kinematic playground &#8599;</a></header>
  <main class="physics-layout">
    <section class="physics-stage" aria-label="Recorded G1 physics replay">
      <div id="physics-scene"></div>
      <div class="physics-overlay"><div class="eyebrow">${mocapReplay ? "EXPERIMENT 007 / CMU PHYSICAL TRACKING" : "EXPERIMENT 002 / PRETRAINED INFERENCE"}</div><h1>${mocapReplay ? "Physical steps.<br><em>Tracking drift.</em>" : "From poses<br>to <em>physics.</em>"}</h1><p>Recorded MuJoCo body transforms.<br>Actual pretrained actions. No browser inference.</p></div>
      <span id="physics-status" class="physics-status">LOADING VERIFIED ROLLOUT</span>
      <div class="physics-legend"><span>TEAL / SIMULATED G1</span><span>AMBER / REFERENCE</span></div>
    </section>
    <aside class="physics-details">
      <div class="eyebrow">CPU POLICY CHECK</div><h2>Pretrained, not pretended.</h2>
      <p id="policy-name"></p>
      <p class="physics-note">A 29-action NVIDIA tracker drove a floating-base G1 in CPU MuJoCo. This player shows the saved result, including tracking errors and any falls.</p>
      <label for="physics-case">${mocapReplay ? "CMU motion reference" : "Self-authored reference"}</label><select id="physics-case" disabled></select>
      <label class="ghost-label"><input id="physics-ghost" type="checkbox" checked /> Show reference skeleton</label>
      <div class="physics-summary"><span id="completion">--</span><span id="fall-result">--</span></div>
      <div class="metric"><span>TRIAL JOINT RMSE</span><div><output id="trial-error">--</output><small>rad</small></div><p>Across all 29 joints and control samples</p></div>
      <div class="metric"><span>MEAN CPU INFERENCE</span><div><output id="inference-time">--</output><small>ms</small></div><p>One ONNX Runtime thread</p></div>
      <div class="physics-live"><span>FRAME ROOT HEIGHT <output id="physics-height">--</output></span><span>FRAME JOINT RMS <output id="physics-error">--</output></span><span>CONTACT CONSTRAINTS <output id="physics-contacts">--</output></span></div>
      <div id="mocap-physical-outcome" class="physics-note" hidden></div>
      <p class="physics-note">${mocapReplay
        ? "Original-timing CMU 07/04 tracking, initialized once from a moving reference pose and velocity. Not a start-from-rest controller. All later motion is actual free-base physics. Four verified steps do not erase the failed world-space tracking gates."
        : "Two synthetic references are smoke tests, <strong>not a published benchmark or proof of generalization</strong>. No BONES motion files, unsafe pickle loads, training, or real robot were used."}</p>
      <a href="${replayUrl}" download class="replay-download">Download measured rollout JSON &#8599;</a>
      <a href="${mocapReplay ? "./physics.html" : "./physics.html?replay=cmu"}" class="replay-download">${mocapReplay ? "Original ten-second standing / arm replay" : "Watch actual CMU physical stepping (tracking gates failed)"} &#8599;</a>
      <p id="physics-provenance" class="physics-provenance"></p>
    </aside>
    <footer class="physics-transport"><button id="physics-pause" disabled>Pause replay</button><button id="physics-reset" disabled>Restart</button><label for="physics-scrub">Recorded time</label><input id="physics-scrub" type="range" min="0" max="10" step="0.04" value="0" disabled /><output id="physics-time">0.00 s</output></footer>
  </main>
  <section class="comparison-section" aria-labelledby="comparison-heading">
    <div class="eyebrow">EXPERIMENT 003 / MATCHED CPU COMPARISON</div>
    <h2 id="comparison-heading">Same references. Different outcomes.</h2>
    <p class="comparison-description">Recorded measurements, not live inference. Two controllers, two self-authored references, three conditions, five seconds each. Both use identical physics, PD gains, effort limits and seeded pelvis pushes.</p>
    <p id="comparison-status" role="status">Loading measured comparison...</p>
    <div id="comparison-error" role="alert" hidden></div>
    <div id="comparison-summary" class="comparison-summary"></div>
    <div class="comparison-table-scroll" tabindex="0" aria-label="All twelve measured episodes; scroll horizontally on narrow screens">
      <table id="comparison-table"><caption>Every episode, including falls and early terminations</caption><thead><tr><th>Reference / condition</th><th>Controller</th><th>Completed time</th><th>Outcome</th><th>Joint RMSE</th><th>Mean body error</th><th>Loop wall time</th></tr></thead><tbody></tbody></table>
    </div>
    <p class="comparison-description">Each push is 60 N horizontally for 0.2 s at 1 s and 3 s (PCG64 seeds 17 and 29). Nominal cases have no pushes. A fall does not end a trial. Success here means completion without a detected fall, not accurate tracking. Joint errors can remain small even when the robot falls.</p>
    <p class="comparison-description">Pooled errors include all observed samples, including falls and truncated trials; early terminations are shown explicitly. No full-horizon comparison is implied if any episode is truncated. Twelve synthetic episodes cannot establish statistical superiority, generalization or state-of-the-art performance. Wall times include host load.</p>
    <a href="./physics-comparison.json" download class="replay-download comparison-download">Download complete comparison and fixed protocol &#8599;</a>
  </section>
  <section id="locomotion-section" class="comparison-section" aria-labelledby="locomotion-heading"></section>
  <section id="mocap-section" class="comparison-section" aria-labelledby="mocap-heading"></section>
  <div id="physics-error-banner" role="alert" hidden></div>`;

function node<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element ${id}`);
  return found as T;
}
function fail(error: unknown) {
  console.error(error);
  node("physics-error-banner").hidden = false;
  node("physics-error-banner").textContent = `Physics replay unavailable: ${error instanceof Error ? error.message : String(error)}. Generate the recorded file with research/rollout.py, then rebuild.`;
  node("physics-status").textContent = "REPLAY UNAVAILABLE";
}

async function loadComparison() {
  const response = await fetch("./physics-comparison.json");
  if (!response.ok) throw new Error(`Comparison request failed (${response.status})`);
  const comparison: unknown = await response.json();
  assertComparison(comparison);
  const label = (controller: string) => controller === "onnx" ? "Pretrained ONNX" : "Direct PD";
  for (const summary of comparison.summary) {
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.innerHTML = `<h3>${label(summary.controller)}</h3>
      <dl><div><dt>Completed without fall</dt><dd>${summary.successes} / 6</dd></div>
      <div><dt>Detected falls</dt><dd>${summary.falls} / 6</dd></div>
      <div><dt>Early terminations</dt><dd>${summary.early_terminations} / 6</dd></div>
      <div><dt>Pooled observed joint RMSE</dt><dd>${summary.observed_joint_rmse_rad.toFixed(4)} rad</dd></div>
      <div><dt>Mean observed body error</dt><dd>${(summary.observed_mean_body_error_m * 100).toFixed(2)} cm</dd></div>
      <div><dt>Total valid simulated time</dt><dd>${summary.simulated_seconds.toFixed(2)} s</dd></div>
      <div><dt>Total loop wall time</dt><dd>${summary.wall_seconds.toFixed(2)} s</dd></div></dl>`;
    node("comparison-summary").append(card);
  }
  const body = node<HTMLTableElement>("comparison-table").tBodies[0];
  for (const episode of comparison.episodes) {
    const metrics = episode.metrics;
    const row = body.insertRow();
    row.dataset.controller = episode.controller;
    row.dataset.failed = String(!metrics.completed || metrics.fell);
    const outcomes = [
      metrics.fell ? `FALL at ${metrics.first_fall_time_s?.toFixed(2)} s` : "No detected fall",
      ...(metrics.completed ? [] : [`EARLY: ${episode.termination ?? "incomplete"}`]),
    ];
    const values = [
      `${episode.name === "stand" ? "Stand" : "Arm sweep"} / ${episode.condition.replace("_", " ")}`,
      label(episode.controller), `${metrics.simulated_seconds.toFixed(2)} / 5 s`, outcomes.join("; "),
      `${metrics.joint_rmse_rad.toFixed(4)} rad`, `${(metrics.mean_body_position_error_m * 100).toFixed(2)} cm`,
      `${metrics.wall_seconds.toFixed(2)} s`,
    ];
    for (const value of values) row.insertCell().textContent = value;
  }
  const early = comparison.summary.reduce((total, summary) => total + summary.early_terminations, 0);
  node("comparison-status").textContent = `12 recorded episodes / ${early} early terminations / fixed protocol ${comparison.protocolSha256.slice(0, 12)}`;
}

async function start() {
  const response = await fetch(replayUrl);
  if (!response.ok) throw new Error(`Rollout request failed (${response.status})`);
  const payload: unknown = await response.json();
  assertReplay(payload);
  const replay: Replay = payload;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#0c151c");
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  const host = node("physics-scene");
  host.append(renderer.domElement);
  renderer.domElement.setAttribute("aria-label", "Recorded floating-base G1 physics and reference skeleton");
  renderer.domElement.setAttribute("role", "img");
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 40);
  camera.position.set(2.6, 1.9, 3.4);
  const controls = new OrbitControls(camera, renderer.domElement);
  let needsRender = true;
  const invalidate = () => { needsRender = true; };
  controls.addEventListener("change", invalidate);
  for (const event of ["input", "change", "click"]) app.addEventListener(event, invalidate);
  controls.target.set(0, 0.7, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 1.8;
  controls.maxDistance = 8;
  scene.add(new THREE.HemisphereLight("#dafef2", "#142329", 3));
  const light = new THREE.DirectionalLight("#ffffff", 2);
  light.position.set(2, 5, 3);
  scene.add(light);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshStandardMaterial({ color: "#172b33", roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.02;
  scene.add(floor);
  const grid = new THREE.GridHelper(20, 40, "#567577", "#31454e");
  scene.add(grid);
  const cylinder = new THREE.CylinderGeometry(0.025, 0.025, 1, 10);
  const sphere = new THREE.SphereGeometry(0.037, 12, 8);
  const mint = new THREE.MeshStandardMaterial({ color: "#a1f1cd", metalness: 0.35, roughness: 0.3 });
  const joints = replay.bodyNames.map((_, i) => {
    const joint = new THREE.Mesh(sphere, mint);
    if (i === 1) joint.scale.setScalar(2.4);
    scene.add(joint);
    return joint;
  });
  const bones = replay.parents.map(parent => {
    if (parent < 0) return null;
    const bone = new THREE.Mesh(cylinder, mint);
    scene.add(bone);
    return bone;
  });
  const ghostPositions = new Float32Array((replay.parents.length - 1) * 6);
  const ghostGeometry = new THREE.BufferGeometry();
  ghostGeometry.setAttribute("position", new THREE.BufferAttribute(ghostPositions, 3));
  const ghost = new THREE.LineSegments(ghostGeometry, new THREE.LineBasicMaterial({ color: "#ffbd69", transparent: true, opacity: 0.8, depthTest: false }));
  ghost.renderOrder = 3;
  scene.add(ghost);
  const point = (p: Point) => new THREE.Vector3(p[0], p[2], -p[1]);
  const up = new THREE.Vector3(0, 1, 0);
  let trial = replay.cases[0], time = 0, paused = false, rendered = false;
  const select = node<HTMLSelectElement>("physics-case");
  for (const item of replay.cases) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.name === "stand" ? "Grounded nominal stand"
      : item.name === "arm_sweep" ? "Slow symmetric arm sweep" : "CMU 07/04: physical stepping, tracking failed";
    select.append(option);
  }
  function setSummary() {
    node("trial-error").textContent = Number(trial.metrics.joint_rmse_rad).toFixed(4);
    node("inference-time").textContent = Number(trial.metrics.mean_inference_ms).toFixed(2);
    node("completion").textContent = trial.metrics.completed ? `${Number(trial.metrics.simulated_seconds).toFixed(1)} s completed` : "Incomplete rollout";
    node("fall-result").textContent = trial.metrics.fell ? "FALL DETECTED" : "No fall detected";
    node("fall-result").classList.toggle("warning", Boolean(trial.metrics.fell));
    node<HTMLInputElement>("physics-scrub").max = String(trial.frames.at(-1)!.time);
  }
  select.addEventListener("change", () => { trial = replay.cases.find(item => item.name === select.value)!; time = 0; setSummary(); });
  node("policy-name").textContent = replay.policy;
  if (replay.locomotion) {
    const outcome = node("mocap-physical-outcome"), m = replay.locomotion.metrics;
    outcome.hidden = false;
    const title = document.createElement("strong");
    title.id = "mocap-physical-verdict";
    title.textContent = replay.locomotion.passed ? "ALL LOCAL LOCOMOTION GATES PASSED" : "REAL STEPS RECORDED / TRAJECTORY GATES FAILED";
    outcome.append(title);
    for (const text of [
      `Verified steps left / right: ${m.verified_steps_per_foot.join(" / ")}.`,
      `Forward progress: ${m.forward_progress_m.toFixed(3)} m. Lateral drift: ${m.final_lateral_drift_m.toFixed(3)} m.`,
      `Root XY RMSE: ${m.root_xy_rmse_m.toFixed(3)} m (limit 0.150 m).`,
      `Mean foot slip: ${m.mean_contact_slip_m_s.map(v => (v * 1000).toFixed(2)).join(" / ")} mm/s.`,
      `Failed gates: ${Object.entries(replay.locomotion.gates).filter(([, passed]) => !passed).map(([name]) => name.replaceAll("_", " ")).join(", ") || "none"}.`,
    ]) {
      const p = document.createElement("p");
      p.textContent = text;
      outcome.append(p);
    }
  }
  node("physics-provenance").textContent = `Source ${replay.provenance.revision.slice(0, 12)} / recorded ${replay.createdAt.slice(0, 10)}. Model: NVIDIA Open Model License. Robot: Unitree BSD-3-Clause.`;
  node<HTMLInputElement>("physics-ghost").addEventListener("change", event => { ghost.visible = (event.target as HTMLInputElement).checked; });
  function setPaused(value: boolean) {
    paused = value;
    node("physics-pause").textContent = paused ? "Play replay" : "Pause replay";
  }
  node("physics-pause").addEventListener("click", () => setPaused(!paused));
  node("physics-reset").addEventListener("click", () => { time = 0; });
  node<HTMLInputElement>("physics-scrub").addEventListener("input", event => {
    time = Number((event.target as HTMLInputElement).value);
    setPaused(true);
  });
  for (const id of ["physics-case", "physics-pause", "physics-reset", "physics-scrub"]) node<HTMLButtonElement>(id).disabled = false;
  function resize() {
    renderer.setSize(host.clientWidth, host.clientHeight);
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
    invalidate();
  }
  new ResizeObserver(resize).observe(host);
  resize();
  setSummary();
  let last: number | undefined, running = true;
  renderer.domElement.addEventListener("webglcontextlost", event => {
    event.preventDefault();
    running = false;
    rendered = false;
    fail(new Error("WebGL context lost; reload to restart"));
  });
  window.physicsReplay = { snapshot: () => ({ ready: rendered, case: trial.name, time, paused, kind: replay.kind, frameCount: trial.frames.length }) };
  const previousFollow = new THREE.Vector3(0, .7, 0);
  function animate(now: number) {
    if (!running) return;
    try {
      const elapsed = last === undefined ? 0 : (now - last) / 1000;
      last = now;
      const duration = trial.frames.at(-1)!.time;
      if (!paused) time = (time + Math.min(elapsed, 0.1)) % duration;
      const index = Math.min(trial.frames.length - 1, Math.floor(time / duration * (trial.frames.length - 1)));
      const frame = trial.frames[index];
      if (mocapReplay) {
        const follow = new THREE.Vector3((frame.body[0][0] + frame.reference[0][0]) / 2, .7,
          -(frame.body[0][1] + frame.reference[0][1]) / 2);
        const delta = follow.clone().sub(previousFollow);
        camera.position.add(delta);
        controls.target.add(delta);
        previousFollow.copy(follow);
      }
      let offset = 0;
      frame.body.forEach((p, i) => {
        const current = point(p);
        joints[i].position.copy(current);
        const parent = replay.parents[i];
        const bone = bones[i];
        if (parent >= 0 && bone) {
          const start = point(frame.body[parent]);
          const delta = current.clone().sub(start);
          bone.position.copy(start).add(current).multiplyScalar(0.5);
          bone.scale.y = delta.length();
          bone.visible = delta.length() > 1e-6;
          if (bone.visible) bone.quaternion.setFromUnitVectors(up, delta.normalize());
          const a = point(frame.reference[parent]), b = point(frame.reference[i]);
          ghostPositions.set([...a.toArray(), ...b.toArray()], offset);
          offset += 6;
        }
      });
      ghostGeometry.attributes.position.needsUpdate = true;
      ghostGeometry.computeBoundingSphere();
      controls.update();
      if (!paused || needsRender) {
        renderer.render(scene, camera);
        needsRender = false;
        rendered = true;
      }
      node("physics-status").textContent = `RECORDED PHYSICS / ${paused ? "PAUSED" : "PLAYING"}`;
      node("physics-height").textContent = `${frame.rootHeightM.toFixed(3)} m`;
      node("physics-error").textContent = `${frame.jointRmsRad.toFixed(4)} rad`;
      node("physics-contacts").textContent = String(frame.contacts);
      node("physics-time").textContent = `${time.toFixed(2)} s`;
      node<HTMLInputElement>("physics-scrub").value = String(time);
      requestAnimationFrame(animate);
    } catch (error) { running = false; rendered = false; fail(error); }
  }
  requestAnimationFrame(animate);
}

declare global {
  interface Window {
    physicsReplay?: { snapshot: () => { ready: boolean; case: string; time: number; paused: boolean; kind: string; frameCount: number } };
  }
}
start().catch(fail);
void loadLocomotion(node("locomotion-section"));
void loadMocap(node("mocap-section"));
loadComparison().catch(error => {
  console.error(error);
  node("comparison-status").textContent = "COMPARISON UNAVAILABLE";
  node("comparison-error").hidden = false;
  node("comparison-error").textContent = `Measured comparison unavailable: ${error instanceof Error ? error.message : String(error)}. The ten-second replay is separate; no comparison results have been invented.`;
});
