import { test, expect, type Page } from "@playwright/test";

async function waitForReplay(page: Page) {
  await expect.poll(() => page.evaluate(() => window.physicsReplay?.snapshot().ready), { timeout: 60000 }).toBe(true);
}

test("physics replay displays recorded policy data and supports case, pause and scrubbing", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/physics.html");
  await waitForReplay(page);
  await expect(page.locator("#physics-status")).toContainText("RECORDED PHYSICS");
  const snapshot = () => page.evaluate(() => window.physicsReplay!.snapshot());
  expect((await snapshot()).kind).toBe("recorded-pretrained-policy-physics");
  expect((await snapshot()).frameCount).toBeGreaterThan(20);
  await page.locator("#physics-pause").click();
  await expect(page.locator("#comparison-status")).toContainText("12 recorded episodes / 0 early terminations");
  await expect(page.locator("#comparison-table tbody tr")).toHaveCount(12);
  await expect(page.locator('#comparison-table tbody tr[data-controller="onnx"]')).toHaveCount(6);
  await expect(page.locator('#comparison-table tbody tr[data-controller="direct_pd"]')).toHaveCount(6);
  await expect(page.locator('#comparison-table tbody tr[data-failed="true"]')).toHaveCount(6);
  await expect(page.locator("#comparison-summary")).toContainText("0.1549 rad");
  await expect(page.locator("#comparison-summary")).toContainText("0.0989 rad");
  await expect(page.locator("#locomotion-status")).toHaveText("NO PHYSICAL WALKING ACHIEVED / 2 ATTEMPTS RECORDED");
  const progress = page.locator("#locomotion-table tr").filter({ hasText: "Actual forward progress" });
  await expect(progress).toContainText("-2.52 cm");
  await expect(progress).toContainText("-2.37 cm");
  const steps = page.locator("#locomotion-table tr").filter({ hasText: "Verified steps" });
  await expect(steps.locator("td").nth(1)).toHaveText("0 / 0");
  await expect(steps.locator("td").nth(2)).toHaveText("0 / 0");
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
  await page.locator(".comparison-section").first().screenshot({ path: "test-results/kinetic-comparison.png" });
  await page.locator("#locomotion-section").screenshot({ path: "test-results/kinetic-locomotion.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator("#physics-reset").click();
  expect((await snapshot()).time).toBe(0);
  await expect(page.locator("#comparison-heading")).toBeVisible();
  expect(errors).toEqual([]);
});

test("missing physics data is an explicit error rather than a synthetic fallback", async ({ page }) => {
  await page.route("**/physics-replay.json", route => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/physics.html");
  await expect(page.getByRole("alert")).toContainText("Rollout request failed (404)");
  await expect(page.locator("#physics-status")).toHaveText("REPLAY UNAVAILABLE");
});

test("missing comparison does not disable the independent ten-second replay", async ({ page }) => {
  await page.route("**/physics-comparison.json", route => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/physics.html");
  await expect(page.locator("#comparison-error")).toContainText("Comparison request failed (404)");
  await expect(page.locator("#comparison-status")).toHaveText("COMPARISON UNAVAILABLE");
  await waitForReplay(page);
  await page.locator("#physics-pause").click();
  await expect(page.locator("#comparison-table tbody tr")).toHaveCount(0);
});

test("unfinished comparison is rejected instead of displaying partial wins", async ({ page }) => {
  await page.route("**/physics-comparison.json", route => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ schemaVersion: 1, kind: "matched-controller-comparison", status: "running" }),
  }));
  await page.goto("/physics.html");
  await expect(page.locator("#comparison-error")).toContainText("Comparison is incomplete");
  await expect(page.locator("#comparison-table tbody tr")).toHaveCount(0);
  await waitForReplay(page);
  await page.locator("#physics-pause").click();
});

test("missing locomotion evidence does not masquerade as a successful walk", async ({ page }) => {
  await page.route("**/locomotion/summary.json", route => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/physics.html");
  await expect(page.locator("#locomotion-status")).toHaveText("LOCOMOTION EVIDENCE UNAVAILABLE");
  await expect(page.locator("#locomotion-error")).toContainText("Locomotion request failed (404)");
  await expect(page.locator("#locomotion-table tbody tr")).toHaveCount(0);
  await expect(page.locator("#comparison-status")).toContainText("12 recorded episodes");
  await waitForReplay(page);
  await page.locator("#physics-pause").click();
});
