import { test, expect } from "@playwright/test";

test("renders geometry, exercises every mode, pause/reset, steering and joint overlay", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  await page.goto("/");
  await expect.poll(() => page.evaluate(() => window.kinetic?.snapshot().rendererReady)).toBe(true);
  const snapshot = () => page.evaluate(() => window.kinetic!.snapshot());
  expect((await snapshot()).triangles).toBeGreaterThan(1000);
  for (const mode of ["idle", "walk", "jog", "march"] as const) {
    await page.locator(`[data-mode="${mode}"]`).click();
    await expect.poll(async () => (await snapshot()).mode).toBe(mode);
    expect((await snapshot()).contactErrorM).toBeLessThan(0.0001);
  }
  await page.getByRole("button", { name: "Pause simulation" }).click();
  const paused = await snapshot();
  await page.waitForTimeout(200);
  expect((await snapshot()).time).toBe(paused.time);
  await page.locator("#target-x").press("End");
  await page.locator("#target-y").press("End");
  await expect(page.locator("#reach-state")).toHaveText("Outside arm workspace");
  expect((await snapshot()).reachErrorM).toBeGreaterThan(0.1);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  expect((await snapshot()).position).toEqual({ x: 0, y: 0, z: 0 });
  expect((await snapshot()).time).toBe(0);
  await page.locator("#skeleton").check();
  expect((await snapshot()).overlay).toBe(true);
  await page.locator("#reach").uncheck();
  await expect(page.locator("#reach-state")).toHaveText("IK disabled / arm swing");
  await page.locator("#reach").check();
  await page.locator('[data-mode="walk"]').click();
  await page.locator("#speed").press("End");
  await page.locator("#turn").press("End");
  await page.getByRole("button", { name: "Resume simulation" }).click();
  await expect.poll(async () => (await snapshot()).heading).toBeGreaterThan(0.03);
  await page.locator("#center-turn").click();
  await page.getByRole("button", { name: "Pause simulation" }).click();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await page.locator("#skeleton").uncheck();
  for (let i = 0; i < 10; i++) await page.locator("#speed").press("ArrowLeft");
  await page.screenshot({ path: "test-results/kinetic-desktop.png", fullPage: true });
  expect(errors).toEqual([]);
});

test("target can be dragged, snapshots exported and keyboard pause used", async ({ page }) => {
  await page.goto("/");
  await expect.poll(() => page.evaluate(() => window.kinetic?.snapshot().rendererReady)).toBe(true);
  await page.getByRole("button", { name: "Pause simulation" }).click();
  const before = await page.evaluate(() => window.kinetic!.snapshot());
  await page.mouse.move(before.targetScreen.x, before.targetScreen.y);
  await page.mouse.down();
  await page.mouse.move(before.targetScreen.x + 45, before.targetScreen.y - 25, { steps: 5 });
  await page.mouse.up();
  const after = await page.evaluate(() => window.kinetic!.snapshot());
  expect(Math.abs(after.target.x - before.target.x) + Math.abs(after.target.y - before.target.y)).toBeGreaterThan(0.02);
  const downloadEvent = page.waitForEvent("download");
  await page.locator("#export").click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe("kinetic-snapshot.json");
  await page.locator("h1").click();
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Pause simulation" })).toBeVisible();
});

test("mobile layout remains usable without horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect.poll(() => page.evaluate(() => window.kinetic?.snapshot().rendererReady)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('[data-mode="march"]').click();
  await expect(page.locator('[data-mode="march"]')).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Pause simulation" }).click();
  await page.screenshot({ path: "test-results/kinetic-mobile.png", fullPage: true });
});
