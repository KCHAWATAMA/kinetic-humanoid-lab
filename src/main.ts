import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { HumanoidSimulation, clamp, vec } from "./kinematics";
import type { MotionMode, Vec3 } from "./kinematics";
import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <header class="topbar">
    <a class="brand" href="./" aria-label="Kinetic home"><span class="brand-mark">K</span> KINETIC<span class="brand-divider">/</span><span class="brand-sub">HUMANOID LAB</span></a>
    <div class="top-status"><span class="status-dot"></span> LOCAL SIMULATION <span class="version">V 0.1</span></div>
  </header>
  <main class="workspace">
    <aside class="control-panel">
      <div class="eyebrow">EXPERIMENT 001 <span>PROCEDURAL MOTION</span></div>
      <h1>Motion, made<br><em>tangible.</em></h1>
      <p class="intro">A virtual body. An open playground.<br>Explore movement, one joint at a time.</p>
      <section aria-labelledby="motion-title">
        <div class="section-heading"><h2 id="motion-title">Locomotion</h2><span>01</span></div>
        <div class="mode-grid" role="group" aria-label="Motion mode">
          <button data-mode="idle" aria-pressed="false"><span class="mode-icon">|</span>Idle<small>RESTING POSE</small></button>
          <button data-mode="walk" aria-pressed="true"><span class="mode-icon">/</span>Walk<small>STEADY STRIDE</small></button>
          <button data-mode="jog" aria-pressed="false"><span class="mode-icon">//</span>Jog<small>QUICK CADENCE</small></button>
          <button data-mode="march" aria-pressed="false"><span class="mode-icon">^</span>March<small>IN PLACE</small></button>
        </div>
        <label class="range-label" for="speed">Tempo <output id="speed-value">1.00x</output></label>
        <input id="speed" type="range" min="0" max="1.5" step="0.05" value="1" />
        <div class="range-ends"><span>STILL</span><span>1.5x</span></div>
        <label class="range-label" for="turn">Steering <output id="turn-value">STRAIGHT</output></label>
        <input id="turn" type="range" min="-1" max="1" step="0.05" value="0" />
        <div class="range-ends"><span>LEFT / A</span><button id="center-turn" class="text-button">CENTER</button><span>RIGHT / D</span></div>
      </section>
      <section class="reach-section" aria-labelledby="reach-title">
        <div class="section-heading"><h2 id="reach-title">Hand target</h2><label class="toggle"><input id="reach" type="checkbox" checked /><span>IK ON</span></label></div>
        <p class="help">Drag the amber target in the scene, or adjust its body-relative position.</p>
        <div class="axis-row"><label for="target-x">X</label><input id="target-x" type="range" min="0.05" max="1.2" step="0.01" value="0.58" /><output id="target-x-value">0.58</output></div>
        <div class="axis-row"><label for="target-y">Y</label><input id="target-y" type="range" min="0.7" max="2.2" step="0.01" value="1.48" /><output id="target-y-value">1.48</output></div>
        <div class="axis-row"><label for="target-z">Z</label><input id="target-z" type="range" min="-0.6" max="0.9" step="0.01" value="0.35" /><output id="target-z-value">0.35</output></div>
      </section>
      <div class="disclosure"><span class="small-dot"></span><p><strong>Kinematics, not physics.</strong> Analytic limb IK and prescribed foot paths. No learned policy, balance simulation, or SOTA claim.</p></div>
    </aside>
    <section class="viewport" aria-label="Interactive 3D humanoid">
      <div id="scene"></div>
      <div class="scene-top"><span class="scene-label"><span class="status-dot"></span> <span id="run-state">SIMULATION LIVE</span></span><span class="scene-id">KH-01 / BIPED</span></div>
      <div class="scene-caption"><span class="eyebrow">KINEMATIC STUDY</span><h2>Meet your<br>virtual counterpart.</h2><p>Self-authored geometry. Real-time articulation.</p></div>
      <div class="view-controls" role="group" aria-label="Camera controls"><button id="camera-iso" title="Reset orbit camera">3/4 view</button><button id="camera-front">Front</button><label><input id="skeleton" type="checkbox" /> Joint overlay</label></div>
      <div class="scene-legend"><span><i class="legend-ground"></i>STANCE CONTACT</span><span><i class="legend-target"></i>HAND TARGET</span></div>
      <div class="scene-bottom"><span>DRAG TO ORBIT <b>+</b> SCROLL TO ZOOM</span><span>SPACE TO PAUSE</span></div>
    </section>
    <aside class="telemetry" aria-label="Live diagnostics">
      <div class="section-heading"><h2>Live telemetry</h2><span class="status-dot"></span></div>
      <div class="metric primary"><span>RENDER RATE</span><div><output id="fps">--</output><small>FPS</small></div><p>Measured in this browser</p></div>
      <div class="metric"><span>STANCE HEIGHT ERROR</span><div><output id="contact-error">0.000</output><small>mm</small></div><p>Endpoint vs. flat ground</p></div>
      <div class="metric"><span>HAND TARGET ERROR</span><div><output id="reach-error">0.0</output><small>mm</small></div><p id="reach-state">Within reach</p></div>
      <div class="metric"><span>ROOT TRAVEL</span><div><output id="travel">0.00</output><small>m</small></div><p>Displacement from origin</p></div>
      <div class="contacts"><span class="mini-heading">FOOT CONTACTS</span><div><span id="contact-left">L <b>STANCE</b></span><span id="contact-right">R <b>STANCE</b></span></div></div>
      <div class="telemetry-note"><span class="eyebrow">THE MODEL</span><p>4 analytic two-bone chains<br>Fixed segment lengths<br>Flat-ground gait generator<br>No external model weights</p></div>
      <button id="export" class="export-button">Export snapshot <span>&#8599;</span></button>
    </aside>
    <footer class="transport">
      <div class="transport-buttons"><button id="pause" class="pause-button" aria-label="Pause simulation"><span id="pause-icon">II</span></button><button id="reset" class="reset-button">Reset</button></div>
      <div class="transport-info"><strong id="motion-label">Walk cycle</strong><span id="time">00:00.00</span></div>
      <div class="phase-track" aria-label="Gait phase"><div class="phase-ticks"></div><div id="phase-fill"></div><div id="phase-head"></div></div>
      <div class="phase-label">CYCLE <output id="phase">0.00</output></div>
      <div class="transport-tag">ANALYTIC IK <span>/</span> LOCAL WEBGL</div>
    </footer>
  </main>
  <div id="error" role="alert" hidden></div>`;

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing interface element: ${id}`);
  return node as T;
}
const sim = new HumanoidSimulation();
const sceneHost = element("scene");
const simulationLabel = element("run-state");
const numeric = (id: string) => element<HTMLInputElement>(id);
const state = { frame: 0, fps: 0, turn: 0, overlay: false, dragging: false, rendererReady: false };

function reportError(error: unknown) {
  console.error(error);
  const banner = element("error");
  banner.hidden = false;
  banner.textContent = `The 3D demo could not continue: ${error instanceof Error ? error.message : String(error)}. WebGL2 is required. Try a current browser with graphics acceleration enabled.`;
  simulationLabel.textContent = "RENDERER STOPPED";
}

function start() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#0c151c");
  scene.fog = new THREE.Fog("#0c151c", 9, 25);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  renderer.domElement.setAttribute("aria-label", "Articulated humanoid 3D scene; use adjacent controls for keyboard interaction");
  renderer.domElement.setAttribute("role", "img");
  renderer.domElement.addEventListener("webglcontextlost", event => {
    event.preventDefault();
    running = false;
    state.rendererReady = false;
    reportError(new Error("Graphics context lost. Reload to restart"));
  });
  sceneHost.append(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(36, 1, 0.05, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 2.6;
  controls.maxDistance = 9;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.target.set(0, 1.0, 0);
  scene.add(new THREE.HemisphereLight("#d3eafa", "#30313a", 2.3));
  const key = new THREE.DirectionalLight("#ffffff", 3);
  key.position.set(3, 5, 4);
  scene.add(key);
  const rim = new THREE.DirectionalLight("#67e5ce", 2);
  rim.position.set(-3, 3, -3);
  scene.add(rim);

  const material = (color: string, metalness = 0.3, roughness = 0.4) => new THREE.MeshStandardMaterial({ color, metalness, roughness });
  const shell = material("#dce5e7", 0.5, 0.3);
  const shellDark = material("#748b96", 0.6, 0.36);
  const graphite = material("#15242d", 0.6, 0.3);
  const teal = material("#65f0cf", 0.4, 0.3);
  const amber = new THREE.MeshBasicMaterial({ color: "#ffbd69" });
  const robot = new THREE.Group();
  scene.add(robot);

  function mesh(geometry: THREE.BufferGeometry, mat: THREE.Material, parent: THREE.Object3D = robot) {
    const object = new THREE.Mesh(geometry, mat);
    parent.add(object);
    return object;
  }
  function box(x: number, y: number, z: number, mat: THREE.Material) { return mesh(new THREE.BoxGeometry(x, y, z), mat); }
  const sphereGeometry = new THREE.SphereGeometry(1, 16, 12);
  function sphere(radius: number, mat: THREE.Material, parent: THREE.Object3D = robot) {
    const object = mesh(sphereGeometry, mat, parent);
    object.scale.setScalar(radius);
    return object;
  }
  const pelvis = box(0.39, 0.17, 0.23, graphite);
  const waist = mesh(new THREE.CylinderGeometry(0.13, 0.14, 0.16, 16), shellDark);
  const torso = mesh(new THREE.CapsuleGeometry(0.225, 0.23, 8, 20), shell);
  torso.scale.set(1, 1, 0.62);
  const chest = box(0.28, 0.16, 0.027, graphite);
  const chestLight = box(0.13, 0.018, 0.015, teal);
  const neck = mesh(new THREE.CylinderGeometry(0.07, 0.085, 0.10, 12), graphite);
  const head = mesh(new THREE.CapsuleGeometry(0.135, 0.095, 6, 18), shell);
  head.scale.z = 0.92;
  const visor = box(0.24, 0.065, 0.068, graphite);
  const eye = box(0.16, 0.012, 0.008, teal);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 14);
  const yAxis = new THREE.Vector3(0, 1, 0);
  function bone(radius: number, mat: THREE.Material) {
    const object = mesh(cylinder, mat);
    object.scale.set(radius, 1, radius);
    return object;
  }
  function place(object: THREE.Object3D, point: Vec3) { object.position.set(point.x, point.y, point.z); }
  function connect(object: THREE.Mesh, a: Vec3, b: Vec3) {
    const delta = new THREE.Vector3(b.x - a.x, b.y - a.y, b.z - a.z);
    object.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
    object.scale.y = delta.length();
    object.quaternion.setFromUnitVectors(yAxis, delta.normalize());
  }
  const chains = Array.from({ length: 4 }, (_, i) => ({
    upper: bone(i < 2 ? 0.087 : 0.060, shell),
    lower: bone(i < 2 ? 0.069 : 0.049, shellDark),
    root: sphere(i < 2 ? 0.102 : 0.085, graphite),
    joint: sphere(i < 2 ? 0.081 : 0.062, graphite),
    end: sphere(i < 2 ? 0.065 : 0.060, i === 3 ? teal : graphite),
  }));
  const feet = [box(0.15, 0.09, 0.29, shell), box(0.15, 0.09, 0.29, shell)];
  const soles = [box(0.156, 0.027, 0.296, graphite), box(0.156, 0.027, 0.296, graphite)];
  const floor = mesh(new THREE.PlaneGeometry(160, 160), material("#101c24", 0.1, 0.96), scene);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.003;
  const grid = new THREE.GridHelper(80, 160, "#34515b", "#233640");
  grid.position.y = 0.001;
  scene.add(grid);
  const platformRing = mesh(new THREE.RingGeometry(1.35, 1.36, 96), new THREE.MeshBasicMaterial({ color: "#365f65", side: THREE.DoubleSide, transparent: true, opacity: 0.6 }));
  platformRing.rotation.x = -Math.PI / 2;
  platformRing.position.y = 0.004;
  const shadow = mesh(new THREE.CircleGeometry(0.47, 40), new THREE.MeshBasicMaterial({ color: "#050c11", transparent: true, opacity: 0.5, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.006;
  shadow.scale.set(1, 0.75, 1);
  const contacts = [0, 1].map(() => {
    const contact = mesh(new THREE.RingGeometry(0.12, 0.14, 32), new THREE.MeshBasicMaterial({ color: "#68e4c4", transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
    contact.rotation.x = -Math.PI / 2;
    return contact;
  });
  const target = sphere(0.055, amber);
  const targetRing = mesh(new THREE.TorusGeometry(0.105, 0.004, 8, 40), amber);
  const targetHalo = sphere(0.11, new THREE.MeshBasicMaterial({ color: "#ffbd69", transparent: true, opacity: 0.09, depthWrite: false }));
  const errorGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const errorLine = new THREE.Line(errorGeometry, new THREE.LineBasicMaterial({ color: "#ffbd69", transparent: true, opacity: 0.7 }));
  robot.add(errorLine);
  const overlayGeometry = new THREE.BufferGeometry();
  const overlayPositions = new Float32Array(4 * 4 * 3);
  overlayGeometry.setAttribute("position", new THREE.BufferAttribute(overlayPositions, 3));
  const overlay = new THREE.LineSegments(overlayGeometry, new THREE.LineBasicMaterial({ color: "#8fffe8", depthTest: false, transparent: true, opacity: 0.9 }));
  overlay.renderOrder = 5;
  robot.add(overlay);

  function cameraView(front: boolean) {
    const p = sim.position;
    camera.position.set(p.x + (front ? 0 : 3.4), front ? 1.7 : 2.6, p.z + (front ? 5.5 : 5.0));
    controls.target.set(p.x, 1.0, p.z);
    controls.update();
  }
  cameraView(false);
  element("camera-iso").addEventListener("click", () => cameraView(false));
  element("camera-front").addEventListener("click", () => cameraView(true));
  const resize = () => {
    const width = sceneHost.clientWidth, height = sceneHost.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(sceneHost);
  resize();
  const syncTarget = () => {
    for (const axis of ["x", "y", "z"] as const) {
      numeric(`target-${axis}`).value = String(sim.target[axis]);
      element(`target-${axis}-value`).textContent = sim.target[axis].toFixed(2);
    }
  };
  for (const axis of ["x", "y", "z"] as const) {
    numeric(`target-${axis}`).addEventListener("input", event => {
      sim.target[axis] = Number((event.target as HTMLInputElement).value);
      syncTarget();
    });
  }
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach(button => button.addEventListener("click", () => {
    sim.mode = button.dataset.mode as MotionMode;
    document.querySelectorAll("[data-mode]").forEach(node => node.setAttribute("aria-pressed", String(node === button)));
    element("motion-label").textContent = `${button.dataset.mode![0].toUpperCase()}${button.dataset.mode!.slice(1)} cycle`;
  }));
  numeric("speed").addEventListener("input", () => {
    sim.speedScale = Number(numeric("speed").value);
    element("speed-value").textContent = `${sim.speedScale.toFixed(2)}x`;
  });
  function setTurn(value: number) {
    state.turn = value;
    numeric("turn").value = String(value);
    element("turn-value").textContent = value === 0 ? "STRAIGHT" : `${Math.abs(value * 100).toFixed(0)}% ${value < 0 ? "LEFT" : "RIGHT"}`;
  }
  numeric("turn").addEventListener("input", () => setTurn(Number(numeric("turn").value)));
  element("center-turn").addEventListener("click", () => setTurn(0));
  numeric("reach").addEventListener("change", () => {
    sim.reaching = numeric("reach").checked;
    document.querySelector(".toggle span")!.textContent = sim.reaching ? "IK ON" : "IK OFF";
  });
  numeric("skeleton").addEventListener("change", () => { state.overlay = numeric("skeleton").checked; });
  function togglePause() {
    sim.paused = !sim.paused;
    element("pause").setAttribute("aria-label", sim.paused ? "Resume simulation" : "Pause simulation");
    element("pause-icon").textContent = sim.paused ? ">" : "II";
    simulationLabel.textContent = sim.paused ? "SIMULATION PAUSED" : "SIMULATION LIVE";
  }
  element("pause").addEventListener("click", togglePause);
  element("reset").addEventListener("click", () => {
    sim.reset();
    setTurn(0);
    previousRoot.copy(new THREE.Vector3());
    cameraView(false);
    syncTarget();
  });
  const keys = new Set<string>();
  window.addEventListener("keydown", event => {
    if (event.target instanceof HTMLElement && event.target.closest("input, button, a, select, textarea")) return;
    if (["Space", "KeyA", "KeyD", "ArrowLeft", "ArrowRight"].includes(event.code)) event.preventDefault();
    if (event.code === "Space" && !event.repeat) togglePause();
    keys.add(event.code);
  });
  window.addEventListener("keyup", event => keys.delete(event.code));
  window.addEventListener("blur", () => keys.clear());

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const dragPlane = new THREE.Plane();
  const dragPoint = new THREE.Vector3();
  function pointerRay(event: PointerEvent) {
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
  }
  renderer.domElement.addEventListener("pointerdown", event => {
    if (!sim.reaching) return;
    pointerRay(event);
    if (!raycaster.intersectObject(targetHalo).length) return;
    state.dragging = true;
    controls.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
    dragPlane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), target.getWorldPosition(new THREE.Vector3()));
    event.stopImmediatePropagation();
  }, true);
  renderer.domElement.addEventListener("pointermove", event => {
    if (!state.dragging) return;
    pointerRay(event);
    if (raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
      const local = robot.worldToLocal(dragPoint.clone());
      sim.target = vec(clamp(local.x, 0.05, 1.2), clamp(local.y, 0.7, 2.2), clamp(local.z, -0.6, 0.9));
      syncTarget();
    }
  });
  function endDrag() { state.dragging = false; controls.enabled = true; }
  renderer.domElement.addEventListener("pointerup", endDrag);
  renderer.domElement.addEventListener("pointercancel", endDrag);
  renderer.domElement.addEventListener("lostpointercapture", endDrag);

  const snapshot = () => {
    const pose = sim.pose();
    const projected = target.getWorldPosition(new THREE.Vector3()).project(camera);
    const bounds = renderer.domElement.getBoundingClientRect();
    return {
      kind: "procedural-kinematics-not-physics", mode: sim.mode, paused: sim.paused,
      time: sim.time, phase: sim.phase % 1, position: { ...sim.position }, heading: sim.heading,
      target: { ...sim.target }, reaching: sim.reaching, speedScale: sim.speedScale,
      contactErrorM: pose.contactError, reachErrorM: pose.reachError,
      contacts: pose.legs.map(leg => leg.contact), rendererReady: state.rendererReady,
      frame: state.frame, fps: state.fps, overlay: state.overlay,
      targetScreen: { x: bounds.left + (projected.x + 1) * bounds.width / 2, y: bounds.top + (1 - projected.y) * bounds.height / 2 },
      drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles,
    };
  };
  window.kinetic = { snapshot };
  element("export").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot(), null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "kinetic-snapshot.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  let last: number | undefined;
  let lastTelemetry = 0, fpsStart = 0, fpsFrames = 0;
  const previousRoot = new THREE.Vector3();
  let running = true;
  function animate(now: number) {
    if (!running) return;
    try {
      const dt = last === undefined ? 0 : (now - last) / 1000;
      if (last === undefined) fpsStart = now;
      last = now;
      const turn = (keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0) - (keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0);
      sim.step(dt, turn || state.turn);
      const root = new THREE.Vector3(sim.position.x, 0, sim.position.z);
      const delta = root.clone().sub(previousRoot);
      camera.position.add(delta);
      controls.target.add(delta);
      previousRoot.copy(root);
      robot.position.copy(root);
      robot.rotation.y = sim.heading;
      floor.position.x = root.x;
      floor.position.z = root.z;
      grid.position.x = Math.round(root.x / 10) * 10;
      grid.position.z = Math.round(root.z / 10) * 10;
      const pose = sim.pose();
      pelvis.position.y = pose.hipHeight;
      waist.position.y = pose.hipHeight + 0.13;
      torso.position.y = pose.hipHeight + 0.37;
      chest.position.set(0, pose.hipHeight + 0.42, 0.14);
      chestLight.position.set(0, pose.hipHeight + 0.42, 0.16);
      neck.position.y = pose.hipHeight + 0.65;
      head.position.y = pose.hipHeight + 0.82;
      visor.position.set(0, pose.hipHeight + 0.84, 0.103);
      eye.position.set(0, pose.hipHeight + 0.84, 0.14);
      const limbs = [...pose.legs, ...pose.arms];
      limbs.forEach((limb, i) => {
        const chain = chains[i];
        connect(chain.upper, limb.root, limb.joint);
        connect(chain.lower, limb.joint, limb.end);
        place(chain.root, limb.root);
        place(chain.joint, limb.joint);
        place(chain.end, limb.end);
        [limb.root, limb.joint, limb.joint, limb.end].forEach((point, j) => {
          overlayPositions.set([point.x, point.y, point.z], i * 12 + j * 3);
        });
      });
      overlayGeometry.attributes.position.needsUpdate = true;
      overlayGeometry.computeBoundingSphere();
      overlay.visible = state.overlay;
      pose.legs.forEach((leg, i) => {
        feet[i].position.set(leg.end.x, leg.end.y - 0.038, leg.end.z + 0.06);
        soles[i].position.set(leg.end.x, leg.end.y - 0.0765, leg.end.z + 0.06);
        contacts[i].position.set(leg.end.x, 0.008, leg.end.z + 0.04);
        contacts[i].visible = leg.contact;
      });
      for (const object of [target, targetRing, targetHalo]) {
        place(object, sim.target);
        object.visible = sim.reaching;
      }
      targetRing.quaternion.copy(camera.quaternion).premultiply(robot.quaternion.clone().invert());
      const errorPositions = errorGeometry.attributes.position;
      errorPositions.setXYZ(0, pose.arms[1].end.x, pose.arms[1].end.y, pose.arms[1].end.z);
      errorPositions.setXYZ(1, sim.target.x, sim.target.y, sim.target.z);
      errorPositions.needsUpdate = true;
      errorGeometry.computeBoundingSphere();
      errorLine.visible = sim.reaching && pose.reachError > 0.001;
      controls.update();
      renderer.render(scene, camera);
      state.rendererReady = true;
      state.frame++;
      fpsFrames++;
      if (now - fpsStart >= 1000) {
        state.fps = fpsFrames * 1000 / (now - fpsStart);
        fpsFrames = 0;
        fpsStart = now;
      }
      if (now - lastTelemetry > 100) {
        element("fps").textContent = state.fps ? state.fps.toFixed(0) : "--";
        element("contact-error").textContent = (pose.contactError * 1000).toFixed(3);
        element("reach-error").textContent = (pose.reachError * 1000).toFixed(1);
        element("reach-state").textContent = !sim.reaching ? "IK disabled / arm swing" : pose.reachError < 0.001 ? "Within reach" : "Outside arm workspace";
        element("reach-state").classList.toggle("warning", sim.reaching && pose.reachError >= 0.001);
        element("travel").textContent = Math.hypot(root.x, root.z).toFixed(2);
        pose.legs.forEach((leg, i) => {
          const contact = element(i === 0 ? "contact-left" : "contact-right");
          contact.innerHTML = `${i === 0 ? "L" : "R"} <b>${leg.contact ? "STANCE" : "SWING"}</b>`;
          contact.classList.toggle("swing", !leg.contact);
        });
        const phase = sim.phase % 1;
        element("phase-fill").style.width = `${phase * 100}%`;
        element("phase-head").style.left = `${phase * 100}%`;
        element("phase").textContent = phase.toFixed(2);
        element("time").textContent = `${Math.floor(sim.time / 60).toString().padStart(2, "0")}:${(sim.time % 60).toFixed(2).padStart(5, "0")}`;
        lastTelemetry = now;
      }
      requestAnimationFrame(animate);
    } catch (error) {
      running = false;
      state.rendererReady = false;
      reportError(error);
    }
  }
  requestAnimationFrame(animate);
}

declare global {
  interface Window {
    kinetic?: { snapshot: () => {
      kind: string; mode: MotionMode; paused: boolean; time: number; phase: number;
      position: Vec3; heading: number; target: Vec3; reaching: boolean; speedScale: number;
      contactErrorM: number; reachErrorM: number; contacts: boolean[]; rendererReady: boolean;
      frame: number; fps: number; overlay: boolean; targetScreen: { x: number; y: number };
      drawCalls: number; triangles: number;
    } };
  }
}

try { start(); } catch (error) { reportError(error); }
