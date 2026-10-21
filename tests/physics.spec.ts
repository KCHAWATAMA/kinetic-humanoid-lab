import { test, expect } from "@playwright/test";

test("physics replay displays recorded policy data and supports case, pause and scrubbing", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/physics.html");
  await expect.poll(() => page.evaluate(() => window.physicsReplay?.snapshot().ready)).toBe(true);
  await expect(page.locator("#physics-status")).toContainText("RECORDED PHYSICS");
  const snapshot = () => page.evaluate(() => window.physicsReplay!.snapshot());
  expect((await snapshot()).kind).toBe("recorded-pretrained-policy-physics");
  expect((await snapshot()).frameCount).toBeGreaterThan(20);
  await page.locator("#physics-pause").click();
  const paused = await snapshot();
  await page.waitForTimeout(150);
  expect((await snapshot()).time).toBe(paused.time);
  await page.locator("#physics-case").selectOption("arm_sweep");
  expect((await snapshot()).case).toBe("arm_sweep");
  await page.locator("#physics-scrub").press("End");
  await expect.poll(async () => (await snapshot()).time).toBe(10);
  await page.locator("#physics-ghost").uncheck();
  await page.locator("#physics-reset").click();
  expect((await snapshot()).time).toBe(0);
  await expect(page.locator("#completion")).toHaveText("10.0 s completed");
  await expect(page.locator("#trial-error")).not.toHaveText("--");
  await page.locator("#physics-scrub").press("End");
  await page.locator("#physics-ghost").check();
  await expect(page.locator("#physics-time")).toHaveText("10.00 s");
  await page.screenshot({ path: "test-results/kinetic-physics.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator("#physics-reset").click();
  expect((await snapshot()).time).toBe(0);
  expect(errors).toEqual([]);
});

test("missing physics data is an explicit error rather than a synthetic fallback", async ({ page }) => {
  await page.route("**/physics-replay.json", route => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/physics.html");
  await expect(page.getByRole("alert")).toContainText("Rollout request failed (404)");
  await expect(page.locator("#physics-status")).toHaveText("REPLAY UNAVAILABLE");
});
