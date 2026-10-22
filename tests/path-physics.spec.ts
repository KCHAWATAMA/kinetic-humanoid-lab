import { expect, test } from "@playwright/test";

test("path comparison switches actual baseline and new feedback with unchanged failed gates", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/physics.html?replay=path");
  await expect.poll(() => page.evaluate(() => window.physicsReplay?.snapshot().ready), { timeout: 60000 }).toBe(true);
  await page.locator("#physics-pause").click();
  await expect(page.locator("#physics-case")).toHaveValue("heading_feedback");
  await expect(page.locator("#path-controller-note")).toContainText("NEW EXTERNAL FEEDBACK");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Root XY RMSE: 0.283 m");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Lateral drift: -0.171 m");
  await expect(page.locator("#mocap-physical-verdict")).toContainText("TRAJECTORY GATES FAILED");
  await page.locator("#physics-scrub").press("End");
  await expect(page.locator("#physics-time")).toHaveText("3.72 s");
  await page.locator(".physics-layout").screenshot({ path: "test-results/kinetic-path-physics.png" });
  await page.locator("#physics-case").selectOption("baseline");
  await expect(page.locator("#path-controller-note")).toContainText("UNMODIFIED PRETRAINED BASELINE");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Root XY RMSE: 0.466 m");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Lateral drift: -0.777 m");
  await expect.poll(() => page.evaluate(() => window.physicsReplay!.snapshot().time)).toBe(0);
  await page.locator("#physics-case").selectOption("heading_feedback");
  await page.locator("#physics-ghost").uncheck();
  await page.locator("#physics-reset").click();
  await expect.poll(() => page.evaluate(() => window.physicsReplay!.snapshot().case)).toBe("heading_feedback");
  const table = page.locator("#path-comparison-table");
  await expect(table.locator("tbody tr")).toHaveCount(11);
  await expect(table.locator("tr").filter({ hasText: "Root XY RMSE" })).toContainText("0.466 m / FAILED");
  await expect(table.locator("tr").filter({ hasText: "Root XY RMSE" })).toContainText("0.283 m / FAILED");
  await expect(table.locator("tr").filter({ hasText: "along-track" })).toContainText("-0.304 m");
  await page.locator("#path-comparison-section").screenshot({ path: "test-results/kinetic-path-comparison.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("missing path comparison never falls back to an uncorrected or synthetic replay", async ({ page }) => {
  await page.route("**/path-tracking-v1/physics-replay.json", route => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/physics.html?replay=path");
  await expect(page.locator("#physics-status")).toHaveText("REPLAY UNAVAILABLE");
  await expect(page.locator("#physics-error-banner")).toContainText("Rollout request failed (404)");
  await expect(page.locator("#path-comparison-section")).toBeHidden();
});

test("path comparison rejects a missing baseline instead of presenting only the better case", async ({ page }) => {
  await page.route("**/path-tracking-v1/physics-replay.json", async route => {
    const response = await route.fetch();
    const data = await response.json();
    data.cases.shift();
    await route.fulfill({ json: data });
  });
  await page.goto("/physics.html?replay=path");
  await expect(page.locator("#physics-error-banner")).toContainText("requires both measured controller cases");
  await expect(page.locator("#path-comparison-section")).toBeHidden();
});
