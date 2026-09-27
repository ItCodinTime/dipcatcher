/**
 * Smoke-level Playwright checks for the replay viewer:
 *  - fixture session loads, canvas renders non-blank, seek works
 *  - screenshot stored as artifact (no brittle pixel-diff)
 *  - bench mode reports frame stats on a 390-bar x 8-symbol synthetic day
 */

import { test, expect, type Page } from "@playwright/test";

interface BenchResult {
  kind: string;
  symbols: number;
  bars: number;
  frames: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
  fps_equiv: number;
}

/** Count pixels in the charts canvas that differ from the page background. */
async function nonBlankPixels(page: Page): Promise<number> {
  return page.evaluate(() => {
    const cv = document.getElementById("charts") as HTMLCanvasElement;
    const ctx = cv.getContext("2d")!;
    const { data } = ctx.getImageData(0, 0, cv.width, cv.height);
    // background is #0e1117; count pixels far from it
    let n = 0;
    for (let i = 0; i < data.length; i += 16) {
      const r = data[i]!;
      const g = data[i + 1]!;
      const b = data[i + 2]!;
      if (Math.abs(r - 14) + Math.abs(g - 17) + Math.abs(b - 23) > 30) n++;
    }
    return n;
  });
}

test("fixture session renders non-blank and seek updates the view", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.locator("#status")).toContainText("ready", { timeout: 20_000 });
  // let a couple of frames render
  await page.waitForTimeout(300);

  expect(await nonBlankPixels(page)).toBeGreaterThan(500);
  expect(await page.locator("#tape .row").count()).toBeGreaterThan(2);

  // seek to ~70% of the session via the test hook; clock + tape must move
  const t0 = await page.locator("#clock").textContent();
  await page.evaluate(() => {
    (window as unknown as { __replay: { seek: (f: number) => void } }).__replay.seek(0.7);
  });
  await page.waitForTimeout(150);
  const t1 = await page.locator("#clock").textContent();
  expect(t1).not.toBe(t0);
  expect(t1).toMatch(/\d{2}:\d{2}:\d{2}/);
  expect(await page.locator("#tape .row").count()).toBeGreaterThan(10);

  const shot = await page.screenshot({ path: "test-results/replay-viewer.png", fullPage: true });
  await testInfo.attach("replay-viewer", { body: shot, contentType: "image/png" });
});

test("playback toggles and advances the cursor", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#status")).toContainText("ready", { timeout: 20_000 });
  await page.click("#btn-play");
  await expect(page.locator("#btn-play")).toHaveText("⏸");
  const before = await page.locator("#clock").textContent();
  await page.waitForTimeout(800);
  const after = await page.locator("#clock").textContent();
  expect(after).not.toBe(before);
  await page.click("#btn-play");
  await expect(page.locator("#btn-play")).toHaveText("▶");
});

test("bench mode reports frame stats on a 390-bar x 8-symbol synthetic day", async ({ page }, testInfo) => {
  await page.goto("/?bench=1&symbols=8&bars=390&frames=240&seed=7");
  await expect(page.locator("#status")).toContainText("bench done", { timeout: 60_000 });

  const result = await page.evaluate(
    () => (window as unknown as { __benchResult: BenchResult }).__benchResult,
  );
  testInfo.attach("bench-result", { body: JSON.stringify(result, null, 2), contentType: "application/json" });
  console.log(`replay-viz bench: ${JSON.stringify(result)}`);

  expect(result.kind).toBe("replay-viz-bench");
  expect(result.symbols).toBe(8);
  expect(result.bars).toBe(390);
  expect(result.frames).toBe(240);
  // generous bound: a broken render path is an order of magnitude slower;
  // the real numbers are recorded as a test artifact + reported in the PR
  expect(result.mean_ms).toBeLessThan(50);

  const shot = await page.screenshot({ path: "test-results/replay-bench.png" });
  await testInfo.attach("replay-bench", { body: shot, contentType: "image/png" });
});
