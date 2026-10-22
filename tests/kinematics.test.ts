import { test } from "node:test";
import assert from "node:assert/strict";
import { ANKLE_HEIGHT, HumanoidSimulation, LEG_LENGTH, STANCE, distance, footTrajectory, solveTwoBone, vec } from "../src/kinematics.ts";
import type { MotionMode } from "../src/kinematics.ts";

test("two-bone IK preserves both lengths for 10,000 deterministic targets", context => {
  let maxBoneError = 0;
  let maxReachableError = 0;
  for (let i = 0; i < 10000; i++) {
    const t = i * 0.017;
    const root = vec(0.1, 0.8, -0.2);
    const target = vec(Math.sin(t) * 1.3, Math.cos(t * 0.7), Math.sin(t * 1.7));
    const ik = solveTwoBone(root, target, vec(0, 0, 1), 0.48, 0.43);
    maxBoneError = Math.max(maxBoneError, Math.abs(distance(root, ik.joint) - 0.48), Math.abs(distance(ik.joint, ik.end) - 0.43));
    if (ik.reachable) maxReachableError = Math.max(maxReachableError, ik.error);
    assert.ok(Math.abs(distance(root, ik.joint) - 0.48) < 1e-7);
    assert.ok(Math.abs(distance(ik.joint, ik.end) - 0.43) < 1e-7);
    if (ik.reachable) assert.ok(ik.error < 1e-7);
  }
  context.diagnostic(JSON.stringify({ targets: 10000, maxBoneErrorM: maxBoneError, maxReachableErrorM: maxReachableError }));
});

test("coincident and parallel targets remain finite; unreachable targets report residual", () => {
  for (const target of [vec(), vec(0, 0, 0.5), vec(100, 0, 0)]) {
    const ik = solveTwoBone(vec(), target, vec(0, 0, 1), 0.3, 0.3);
    assert.ok(Object.values(ik.joint).every(Number.isFinite));
    assert.ok(Math.abs(distance(ik.joint, ik.end) - 0.3) < 1e-7);
  }
  assert.ok(solveTwoBone(vec(), vec(2, 0, 0), vec(), 0.3, 0.3).error > 1.39);
  assert.throws(() => solveTwoBone(vec(), vec(), vec(), -1, 1), RangeError);
  assert.throws(() => solveTwoBone(vec(), vec(NaN), vec(), 1, 1), RangeError);
});

test("all modes keep bones rigid and stance feet grounded within 0.1 mm", context => {
  let maxContactError = 0;
  for (const mode of ["idle", "walk", "jog", "march"] as MotionMode[]) {
    for (const speed of [0, 0.5, 1, 1.5]) {
      const sim = new HumanoidSimulation();
      sim.mode = mode;
      sim.speedScale = speed;
      for (let frame = 0; frame < 1200; frame++) {
        sim.step(1 / 120);
        const pose = sim.pose();
        maxContactError = Math.max(maxContactError, pose.contactError);
        assert.ok(pose.contactError < 0.0001, `${mode}: contact ${pose.contactError}`);
        for (const leg of pose.legs) {
          assert.ok(Math.abs(distance(leg.root, leg.joint) - LEG_LENGTH) < 1e-7);
          assert.ok(Math.abs(distance(leg.joint, leg.end) - LEG_LENGTH) < 1e-7);
          assert.ok(leg.end.y >= ANKLE_HEIGHT - 0.0001);
        }
      }
    }
  }
  context.diagnostic(JSON.stringify({ samples: 19200, maxContactErrorM: maxContactError }));
});

test("straight-line stance feet remain stationary within 0.1 mm", context => {
  let maxSlide = 0;
  for (const mode of ["walk", "jog"] as MotionMode[]) {
    for (const speed of [0.5, 1, 1.5]) {
      const sim = new HumanoidSimulation();
      sim.mode = mode;
      sim.speedScale = speed;
      let previous = sim.pose();
      let previousPosition = { ...sim.position };
      for (let i = 0; i < 1200; i++) {
        sim.step(1 / 120);
        const next = sim.pose();
        next.legs.forEach((leg, index) => {
          const before = previous.legs[index];
          if (leg.contact && before.contact) {
            const travel = sim.position.z - previousPosition.z;
            maxSlide = Math.max(maxSlide, Math.abs(leg.end.z - before.end.z + travel));
            assert.ok(Math.abs(leg.end.z - before.end.z + travel) < 0.0001);
          }
        });
        previous = next;
        previousPosition = { ...sim.position };
      }
    }
  }
  context.diagnostic(JSON.stringify({ steps: 7200, maxStanceSlidePerStepM: maxSlide }));
});

test("gait trajectory is continuous at stance and cycle boundaries", () => {
  for (const boundary of [0, STANCE, 1]) {
    const before = footTrajectory(boundary - 1e-9, "walk", 1);
    const after = footTrajectory(boundary + 1e-9, "walk", 1);
    assert.ok(Math.hypot(before.y - after.y, before.z - after.z) < 1e-7);
  }
});

test("pause, reset and deterministic simulation", () => {
  const a = new HumanoidSimulation(), b = new HumanoidSimulation();
  for (let i = 0; i < 600; i++) { a.step(1 / 60, 0.2); b.step(1 / 60, 0.2); }
  assert.deepEqual(a.pose(), b.pose());
  a.paused = true;
  const before = JSON.stringify(a);
  a.step(1);
  assert.equal(JSON.stringify(a), before);
  a.reset();
  assert.equal(a.phase, 0);
  assert.deepEqual(a.position, vec());
  assert.throws(() => b.step(-1), RangeError);
});
