import { expect, test } from "@playwright/test";

test("CMU replay shows actual steps and failed tracking without substituting reference motion", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/physics.html?replay=cmu");
  await expect.poll(() => page.evaluate(() => window.physicsReplay?.snapshot().ready), { timeout: 60000 }).toBe(true);
  await page.locator("#physics-pause").click();
  await expect(page.locator("#mocap-physical-verdict")).toHaveText("REAL STEPS RECORDED / TRAJECTORY GATES FAILED");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Verified steps left / right: 2 / 2");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Forward progress: 2.500 m");
  await expect(page.locator("#mocap-physical-outcome")).toContainText("Lateral drift: -0.777 m");
  await expect(page.locator("#fall-result")).toHaveText("No fall detected");
  await expect(page.locator("#physics-case")).toHaveValue("g1-cmu-contact-v2.2");
  await page.locator("#physics-scrub").press("End");
  await expect(page.locator("#physics-time")).toHaveText("3.72 s");
  expect(await page.evaluate(() => window.physicsReplay!.snapshot().frameCount)).toBe(94);
  await page.locator(".physics-layout").screenshot({ path: "test-results/kinetic-cmu-physics.png" });
  await page.locator(".physics-details > a[download]").scrollIntoViewIfNeeded();
  expect(await page.locator(".physics-details > a[download]").evaluate(element => {
    const bounds = element.getBoundingClientRect();
    const sidebar = element.closest(".physics-details")!.getBoundingClientRect();
    return bounds.top >= sidebar.top && bounds.bottom <= sidebar.bottom;
  })).toBe(true);
  await page.locator("#physics-ghost").uncheck();
  await page.locator("#physics-reset").click();
  await expect.poll(() => page.evaluate(() => window.physicsReplay!.snapshot().time)).toBe(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("missing CMU physical data does not fall back to standing or a reference animation", async ({ page }) => {
  await page.route("**/mocap-v2/physics-replay.json", route => route.fulfill({ status: 404, body: "missing" }));
  await page.goto("/physics.html?replay=cmu");
  await expect(page.locator("#physics-status")).toHaveText("REPLAY UNAVAILABLE");
  await expect(page.locator("#physics-error-banner")).toContainText("Rollout request failed (404)");
  await expect(page.locator("#mocap-physical-outcome")).toBeHidden();
});
