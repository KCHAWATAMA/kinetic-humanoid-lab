const metricNames = [
  "source_support_fraction", "maximum_root_reach_lowering_m", "max_foot_position_error_m",
  "max_foot_orientation_error_rad", "max_torso_orientation_error_rad", "max_joint_velocity_rad_s",
  "maximum_stance_slip_m_s", "maximum_ground_penetration_m", "maximum_stance_adjustment_m",
  "tendon_violation_frames", "retarget_frames",
] as const;
type Evidence = {
  reference: { passed: false; gates: Record<string, boolean>; metrics: Record<typeof metricNames[number], number> };
  protocol: { reference_gates: Record<string, number>; retarget: { maximum_root_reach_lowering_m: number } };
};
const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

function assertEvidence(value: unknown): asserts value is Evidence {
  if (!record(value) || value.schemaVersion !== 1 || value.kind !== "cmu-mocap-reference-experiment"
    || value.identity !== "g1-cmu-07-04-v1" || value.status !== "reference_failed" || value.physicsExecuted !== false
    || !record(value.reference) || value.reference.passed !== false || !record(value.reference.gates)
    || Object.keys(value.reference.gates).length !== 12
    || !Object.values(value.reference.gates).every(gate => typeof gate === "boolean")
    || Object.values(value.reference.gates).every(gate => gate === true)
    || !record(value.reference.metrics) || !record(value.protocol) || !record(value.protocol.reference_gates)
    || !record(value.protocol.retarget) || !finite(value.protocol.retarget.maximum_root_reach_lowering_m)) {
    throw new Error("Incomplete or unsupported mocap preflight evidence");
  }
  const metrics = value.reference.metrics, gates = value.protocol.reference_gates;
  if (!metricNames.every(name => finite(metrics[name]) && metrics[name] >= 0)
    || !["minimum_source_support_fraction", "maximum_foot_position_error_m", "maximum_joint_velocity_rad_s",
      "maximum_foot_orientation_error_rad", "maximum_torso_orientation_error_rad", "maximum_stance_slip_m_s",
      "maximum_ground_penetration_m", "maximum_stance_adjustment_m"].every(name => finite(gates[name]) && gates[name] >= 0)) {
    throw new Error("Invalid mocap reference measurements");
  }
}

export async function loadMocap(host: HTMLElement) {
  host.innerHTML = `<div class="eyebrow">HISTORICAL V1 / CMU REFERENCE PREFLIGHT</div>
    <h2 id="mocap-heading">The first CMU retarget failed before physics.</h2>
    <p class="comparison-description">CMU subject 07, trial 04: one 3.72-second slow-walk segment, original timing, no speed sweep. The first G1 retarget failed before policy execution. These preserved v1 diagnostics are not the repaired v2.2 physical trial.</p>
    <a class="replay-download comparison-download" href="./physics.html?replay=cmu">Watch the repaired reference drive actual physical steps; trajectory gates still fail</a>
    <p id="mocap-status" role="status">Loading mocap preflight evidence...</p>
    <div id="mocap-error" role="alert" hidden></div>
    <div class="comparison-table-scroll" tabindex="0" aria-label="CMU reference rejection measurements">
      <table class="experiment-table" id="mocap-table"><caption>Failed reference candidate; no physical trial</caption>
      <thead><tr><th>Reference measurement</th><th>Observed</th><th>Fixed limit</th></tr></thead><tbody></tbody></table>
    </div>
    <p class="comparison-description">The height-based contact estimate rejects a slowly moving right foot late in the clip. Flat-foot adaptation also exceeds the pelvis-reach cap; bounded leg IK hits joint limits, diverges and violates ankle tendons. These are failures of this retargeting approach, not evidence that the pretrained policy cannot walk. Large IK motion errors are not measured physical slip or steps.</p>
    <p class="comparison-description">No thresholds were relaxed and no invalid reference was sent to the policy. One reporting-error recovery used the same protocol; both execution records are retained. The existing standing replay and earlier physical walking failures are separate.</p>
    <p class="comparison-description">The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217. CMU's stated terms apply, not CC-BY; raw ASF/AMC files remain in an ignored local cache.</p>
    <a class="replay-download comparison-download" href="./mocap/g1-cmu-07-04-v1.json" download>Download rejected reference, diagnostics and provenance</a>
    <a class="replay-download comparison-download" href="./mocap/execution-ledger.json" download>Download complete execution ledger</a>`;
  const status = host.querySelector<HTMLElement>("#mocap-status")!;
  try {
    const response = await fetch("./mocap/g1-cmu-07-04-v1.json");
    if (!response.ok) throw new Error(`Mocap evidence request failed (${response.status})`);
    const evidence: unknown = await response.json();
    assertEvidence(evidence);
    const m = evidence.reference.metrics, g = evidence.protocol.reference_gates;
    const rows = [
      ["Inferred source support", `${(m.source_support_fraction * 100).toFixed(2)}%`, `at least ${(g.minimum_source_support_fraction * 100).toFixed(0)}%`],
      ["Required pelvis reach lowering", `${(m.maximum_root_reach_lowering_m * 100).toFixed(2)} cm`, `${(evidence.protocol.retarget.maximum_root_reach_lowering_m * 100).toFixed(0)} cm`],
      ["Maximum foot IK position error", `${(m.max_foot_position_error_m * 1000).toFixed(2)} mm`, `${(g.maximum_foot_position_error_m * 1000).toFixed(1)} mm`],
      ["Maximum foot orientation error", `${m.max_foot_orientation_error_rad.toFixed(4)} rad`, `${g.maximum_foot_orientation_error_rad} rad`],
      ["Maximum torso orientation error", `${m.max_torso_orientation_error_rad.toExponential(2)} rad`, `${g.maximum_torso_orientation_error_rad} rad`],
      ["Maximum reference joint speed", `${m.max_joint_velocity_rad_s.toFixed(2)} rad/s`, `${g.maximum_joint_velocity_rad_s} rad/s`],
      ["Individual joint limits", evidence.reference.gates.joint_limits ? "within limits" : "violated", "within limits"],
      ["Frames violating ankle tendons (including lookahead)", `${m.tendon_violation_frames} / ${m.retarget_frames}`, "0"],
      ["Maximum kinematic stance-foot motion", `${m.maximum_stance_slip_m_s.toFixed(3)} m/s`, `${g.maximum_stance_slip_m_s} m/s`],
      ["Maximum reference floor penetration", `${(m.maximum_ground_penetration_m * 1000).toFixed(3)} mm`, `${g.maximum_ground_penetration_m * 1000} mm`],
      ["Maximum source-foot adjustment", `${(m.maximum_stance_adjustment_m * 100).toFixed(2)} cm`, `${g.maximum_stance_adjustment_m * 100} cm`],
      ["Swing-height gate", evidence.reference.gates.swing_clearance ? "passes height only; invalid IK" : "failed", "not a verified physical step"],
    ];
    const body = host.querySelector<HTMLTableElement>("#mocap-table")!.tBodies[0];
    for (const values of rows) {
      const row = body.insertRow();
      for (const value of values) row.insertCell().textContent = value;
    }
    status.textContent = "REFERENCE REJECTED / 0 POLICY CALLS / 0 PHYSICS STEPS";
  } catch (error) {
    console.error(error);
    status.textContent = "MOCAP EVIDENCE UNAVAILABLE";
    const banner = host.querySelector<HTMLElement>("#mocap-error")!;
    banner.hidden = false;
    banner.textContent = error instanceof Error ? error.message : String(error);
  }
}
