export type Vec3 = { x: number; y: number; z: number };
export type MotionMode = "idle" | "walk" | "jog" | "march";
export const vec = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });
export const add = (a: Vec3, b: Vec3): Vec3 => vec(a.x + b.x, a.y + b.y, a.z + b.z);
export const sub = (a: Vec3, b: Vec3): Vec3 => vec(a.x - b.x, a.y - b.y, a.z - b.z);
export const mul = (a: Vec3, s: number): Vec3 => vec(a.x * s, a.y * s, a.z * s);
export const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
export const distance = (a: Vec3, b: Vec3): number => length(sub(a, b));
export const unit = (a: Vec3): Vec3 => length(a) > 1e-12 ? mul(a, 1 / length(a)) : vec(0, -1, 0);
export const clamp = (x: number, low: number, high: number): number => Math.max(low, Math.min(high, x));

export function solveTwoBone(root: Vec3, target: Vec3, pole: Vec3, upper: number, lower: number) {
  if (!(upper > 0 && lower > 0) || !Number.isFinite(upper + lower)
    || ![root, target, pole].every(v => [v.x, v.y, v.z].every(Number.isFinite))) {
    throw new RangeError("IK requires finite vectors and positive finite bone lengths.");
  }
  const offset = sub(target, root);
  const requestedDistance = length(offset);
  const direction = unit(offset);
  const reach = clamp(requestedDistance, Math.max(1e-8, Math.abs(upper - lower)), upper + lower);
  let bend = sub(pole, mul(direction, dot(pole, direction)));
  if (length(bend) < 1e-8) {
    const axis = Math.abs(direction.x) < 0.8 ? vec(1, 0, 0) : vec(0, 0, 1);
    bend = sub(axis, mul(direction, dot(axis, direction)));
  }
  const along = (upper * upper - lower * lower + reach * reach) / (2 * reach);
  const height = Math.sqrt(Math.max(0, upper * upper - along * along));
  const joint = add(root, add(mul(direction, along), mul(unit(bend), height)));
  const end = add(root, mul(direction, reach));
  return { root, joint, end, error: distance(end, target), reachable: requestedDistance <= upper + lower && requestedDistance >= Math.abs(upper - lower) };
}

const presets = {
  idle: { cadence: 0, speed: 0, lift: 0 },
  walk: { cadence: 1.15, speed: 0.6, lift: 0.13 },
  jog: { cadence: 1.8, speed: 1.05, lift: 0.20 },
  march: { cadence: 1.05, speed: 0, lift: 0.24 },
} as const;

export const STANCE = 0.62;
export const ANKLE_HEIGHT = 0.09;
export const LEG_LENGTH = 0.48;

export function footTrajectory(phase: number, mode: MotionMode, speedScale: number) {
  const preset = presets[mode];
  if (mode === "idle" || speedScale === 0) return { y: ANKLE_HEIGHT, z: 0, contact: true };
  const u = ((phase % 1) + 1) % 1;
  const stride = preset.speed * STANCE / preset.cadence;
  if (u < STANCE) return { y: ANKLE_HEIGHT, z: stride * (0.5 - u / STANCE), contact: true };
  const t = (u - STANCE) / (1 - STANCE);
  const smooth = t * t * (3 - 2 * t);
  return { y: ANKLE_HEIGHT + preset.lift * Math.sin(Math.PI * t) ** 2, z: stride * (smooth - 0.5), contact: false };
}

export class HumanoidSimulation {
  time = 0;
  phase = 0;
  position = vec();
  heading = 0;
  mode: MotionMode = "walk";
  speedScale = 1;
  paused = false;
  target = vec(0.58, 1.48, 0.35);
  reaching = true;

  reset() {
    this.time = 0;
    this.phase = 0;
    this.position = vec();
    this.heading = 0;
    this.target = vec(0.58, 1.48, 0.35);
  }

  step(dt: number, turn = 0) {
    if (!Number.isFinite(dt) || dt < 0 || !Number.isFinite(turn)) throw new RangeError("Time step and turn must be finite; time step must be nonnegative.");
    if (this.paused) return;
    // Bound elapsed time after tab suspension; subdivide turning integration.
    let remaining = Math.min(dt, 0.1);
    while (remaining > 1e-10) {
      const step = Math.min(remaining, 1 / 120);
      const preset = presets[this.mode];
      this.time += step;
      this.heading += turn * step * 1.6;
      this.phase += step * preset.cadence * this.speedScale;
      const travel = preset.speed * this.speedScale * step;
      this.position.x += Math.sin(this.heading) * travel;
      this.position.z += Math.cos(this.heading) * travel;
      remaining -= step;
    }
  }

  world(local: Vec3): Vec3 {
    const c = Math.cos(this.heading), s = Math.sin(this.heading);
    return vec(this.position.x + local.x * c + local.z * s, local.y, this.position.z - local.x * s + local.z * c);
  }

  pose() {
    const active = this.mode !== "idle" && this.speedScale > 0;
    const bob = active ? 0.018 * Math.cos(this.phase * Math.PI * 4) : 0.004 * Math.sin(this.time * 2);
    const hipHeight = 0.98 + bob;
    const legs = ([-1, 1] as const).map(side => {
      const foot = footTrajectory(this.phase + (side === 1 ? 0.5 : 0), this.mode, this.speedScale);
      const target = vec(side * 0.155, foot.y, foot.z);
      return { ...solveTwoBone(vec(side * 0.155, hipHeight, 0), target, vec(0, 0, 1), LEG_LENGTH, LEG_LENGTH), contact: foot.contact, target };
    });
    const shoulders = ([-1, 1] as const).map(side => vec(side * 0.30, hipHeight + 0.52, 0));
    const arms = shoulders.map((root, i) => {
      const side = i === 0 ? -1 : 1;
      const swing = active ? Math.sin(this.phase * Math.PI * 2) * 0.20 * side : 0;
      const target = i === 1 && this.reaching ? this.target : vec(side * 0.36, hipHeight + 0.06, swing);
      return solveTwoBone(root, target, vec(side, -0.1, -1), 0.30, 0.29);
    });
    const contactError = Math.max(...legs.filter(l => l.contact).map(l => Math.abs(l.end.y - ANKLE_HEIGHT)), 0);
    return { hipHeight, legs, arms, contactError, reachError: this.reaching ? arms[1].error : 0 };
  }
}
