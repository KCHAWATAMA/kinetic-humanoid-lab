type CompletedAttempt = {
  attempt: number; status: "complete"; artifact: string; passed: boolean; gates: Record<string, boolean>;
  protocol: { foot_clearance_m: number; stance_root_shift_m: number };
  metrics: {
    forward_progress_m: number; target_forward_progress_m: number; root_xy_rmse_m: number;
    verified_steps_per_foot: number[]; mean_contact_slip_m_s: (number | null)[]; p95_contact_slip_m_s: (number | null)[];
  };
  physicsMetrics: { simulated_seconds: number; fell: boolean; joint_rmse_rad: number; mean_body_position_error_m: number; wall_seconds: number };
};
type FailedReference = { attempt: number; status: "reference_failed"; artifact: string; passed: false; error: string };
type Attempt = CompletedAttempt | FailedReference;
type Summary = { walkingAchieved: boolean; attempts: Attempt[] };
const record = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function assertSummary(value: unknown): asserts value is Summary {
  if (!record(value) || value.schemaVersion !== 1 || value.kind !== "recorded-locomotion-attempts"
    || value.status !== "complete" || typeof value.walkingAchieved !== "boolean"
    || !Array.isArray(value.attempts) || value.attempts.length !== 2) throw new Error("Incomplete or unsupported locomotion evidence");
  for (const [index, attempt] of value.attempts.entries()) {
    if (!record(attempt) || attempt.attempt !== index + 1 || attempt.artifact !== `attempt-0${index + 1}.json`
      || typeof attempt.passed !== "boolean") throw new Error("Invalid locomotion attempt");
    if (attempt.status === "reference_failed") {
      if (attempt.passed || typeof attempt.error !== "string") throw new Error("Invalid reference failure");
      continue;
    }
    if (attempt.status !== "complete" || !record(attempt.metrics) || !record(attempt.physicsMetrics)
      || !record(attempt.protocol) || !record(attempt.gates) || Object.keys(attempt.gates).length !== 12
      || !Object.values(attempt.gates).every(v => typeof v === "boolean")
      || attempt.passed !== Object.values(attempt.gates).every(v => v === true)) throw new Error("Invalid locomotion outcome");
    const metrics = attempt.metrics, physics = attempt.physicsMetrics;
    if (!["forward_progress_m", "target_forward_progress_m", "root_xy_rmse_m"].every(key => finite(metrics[key]))
      || !["simulated_seconds", "joint_rmse_rad", "mean_body_position_error_m", "wall_seconds"].every(key => finite(physics[key]))
      || typeof physics.fell !== "boolean" || !finite(attempt.protocol.foot_clearance_m) || !finite(attempt.protocol.stance_root_shift_m)
      || !Array.isArray(metrics.verified_steps_per_foot) || metrics.verified_steps_per_foot.length !== 2
      || !metrics.verified_steps_per_foot.every(v => Number.isInteger(v) && v >= 0)
      || !["mean_contact_slip_m_s", "p95_contact_slip_m_s"].every(key => {
        const pair = metrics[key];
        return Array.isArray(pair) && pair.length === 2 && pair.every(v => v === null || (finite(v) && v >= 0));
      })) throw new Error("Invalid physical locomotion measurements");
  }
  if (value.walkingAchieved !== value.attempts.some(attempt => attempt.passed)) throw new Error("Locomotion verdict disagrees with attempts");
}

export async function loadLocomotion(host: HTMLElement) {
  host.innerHTML = `<div class="eyebrow">EXPERIMENT 004 / BOUNDED PHYSICAL LOCOMOTION</div>
    <h2 id="locomotion-heading">A walk reference is not a walking robot.</h2>
    <p class="comparison-description">Two recorded eight-second experiments, not a live walking controller. Each used joint-limited G1 leg IK, four alternating foot placements and a +35 cm reference root target. Physical root motion was never assigned from that reference.</p>
    <p id="locomotion-status" role="status">Loading locomotion evidence...</p>
    <div id="locomotion-error" role="alert" hidden></div>
    <div class="comparison-table-scroll" tabindex="0" aria-label="Recorded locomotion attempts">
      <table class="experiment-table" id="locomotion-table"><caption>All attempts; unchanged pass criteria</caption>
      <thead><tr><th>Measurement</th><th>Attempt 01</th><th>Attempt 02</th></tr></thead><tbody></tbody></table>
    </div>
    <p id="locomotion-interpretation" class="comparison-description"></p>
    <p class="comparison-description">Walking requires eight seconds, no fall/warnings, 25-45 cm forward progress, two verified steps per foot, root/body tracking gates and bounded contact slip. A step requires at least 0.10 s unloaded, 2 cm lift and 5 cm forward landing. Foot load is measured from floor-contact forces, not animation flags. Slip is measured at actual contact points at 50 Hz; stationary loaded feet can have tiny slip without walking.</p>
    <p class="comparison-description">Attempt 02 was the final adaptive engineering variant: 8 cm instead of 4 cm clearance, and 10 cm instead of 8 cm stance-side shift. No further tuning or hidden trials. The ONNX export lacks direct root-position and root-linear-velocity targets; these failures alone do not establish why the policy failed or whether compatible motion data would work.</p>
    <div id="locomotion-downloads"></div>`;
  const status = host.querySelector<HTMLElement>("#locomotion-status")!;
  try {
    const response = await fetch("./locomotion/summary.json");
    if (!response.ok) throw new Error(`Locomotion request failed (${response.status})`);
    const summary: unknown = await response.json();
    assertSummary(summary);
    status.textContent = summary.walkingAchieved ? "LOCAL LOCOMOTION GATES PASSED" : "NO PHYSICAL WALKING ACHIEVED / 2 ATTEMPTS RECORDED";
    const pair = (values: (number | null)[]) => values.map(v => v === null ? "N/A" : (v * 1000).toFixed(3)).join(" / ");
    const measurements: [string, (a: CompletedAttempt) => string][] = [
      ["Outcome", a => a.passed ? "Passed local gates" : "FAILED locomotion gates"],
      ["Completed duration / fall", a => `${a.physicsMetrics.simulated_seconds.toFixed(2)} s / ${a.physicsMetrics.fell ? "FALL" : "no fall"}`],
      ["Reference foot clearance / side shift", a => `${(a.protocol.foot_clearance_m * 100).toFixed(0)} / ${(a.protocol.stance_root_shift_m * 100).toFixed(0)} cm`],
      ["Target forward progress", a => `${(a.metrics.target_forward_progress_m * 100).toFixed(2)} cm`],
      ["Actual forward progress", a => `${(a.metrics.forward_progress_m * 100).toFixed(2)} cm`],
      ["Verified steps left / right (need 2 each)", a => a.metrics.verified_steps_per_foot.join(" / ")],
      ["Root XY tracking RMSE (limit 15 cm)", a => `${(a.metrics.root_xy_rmse_m * 100).toFixed(2)} cm`],
      ["Mean body-origin error (limit 15 cm)", a => `${(a.physicsMetrics.mean_body_position_error_m * 100).toFixed(2)} cm`],
      ["Joint RMSE (limit 0.30 rad)", a => `${a.physicsMetrics.joint_rmse_rad.toFixed(4)} rad`],
      ["Mean foot slip left / right (limit 80 mm/s)", a => `${pair(a.metrics.mean_contact_slip_m_s)} mm/s`],
      ["P95 foot slip left / right (limit 200 mm/s)", a => `${pair(a.metrics.p95_contact_slip_m_s)} mm/s`],
      ["Loop wall time", a => `${a.physicsMetrics.wall_seconds.toFixed(2)} s`],
      ["Failed gates", a => Object.entries(a.gates).filter(([, passed]) => !passed).map(([key]) => key.replaceAll("_", " ")).join(", ") || "None"],
    ];
    const body = host.querySelector<HTMLTableElement>("#locomotion-table")!.tBodies[0];
    for (const [name, format] of measurements) {
      const row = body.insertRow();
      row.insertCell().textContent = name;
      for (const attempt of summary.attempts) {
        row.insertCell().textContent = attempt.status === "complete" ? format(attempt) : `Reference failed; no rollout: ${attempt.error}`;
      }
    }
    host.querySelector("#locomotion-interpretation")!.textContent = summary.walkingAchieved
      ? "At least one recorded attempt met the fixed local gates. This is not a published benchmark or SOTA result."
      : "These recorded attempts did not establish physical walking. Low slip, no fall and a successful reference-IK calculation are not enough; all physical progress and step criteria must pass. The original ten-second standing/arm replay above is separate and must not be mistaken for a walking result.";
    for (const attempt of summary.attempts) {
      const link = document.createElement("a");
      link.className = "replay-download comparison-download";
      link.href = `./locomotion/${attempt.artifact}`;
      link.download = attempt.artifact;
      link.textContent = `Download attempt ${attempt.attempt}: configuration, frames, contacts and provenance`;
      host.querySelector("#locomotion-downloads")!.append(link);
    }
  } catch (error) {
    console.error(error);
    status.textContent = "LOCOMOTION EVIDENCE UNAVAILABLE";
    const banner = host.querySelector<HTMLElement>("#locomotion-error")!;
    banner.hidden = false;
    banner.textContent = error instanceof Error ? error.message : String(error);
  }
}
