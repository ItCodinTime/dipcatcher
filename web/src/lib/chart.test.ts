import { describe, expect, it } from "vitest";
import {
  areaPath,
  dateTicks,
  decimateMinMax,
  extent,
  invertScale,
  isoDay,
  linePath,
  linearScale,
  nearestIndex,
  niceTicks,
  paddedExtent,
  parseDay,
} from "./chart";
import type { XY } from "./chart";

describe("linearScale", () => {
  it("maps domain endpoints to range endpoints", () => {
    const s = linearScale(0, 10, 100, 200);
    expect(s(0)).toBe(100);
    expect(s(10)).toBe(200);
    expect(s(5)).toBe(150);
  });

  it("handles inverted ranges (svg y-axis)", () => {
    const s = linearScale(0, 1, 300, 0);
    expect(s(0)).toBe(300);
    expect(s(1)).toBe(0);
  });

  it("degenerate domain maps to range midpoint", () => {
    const s = linearScale(5, 5, 0, 100);
    expect(s(5)).toBe(50);
  });

  it("invertScale round-trips", () => {
    const fwd = linearScale(10, 20, 0, 500);
    const inv = invertScale(0, 500, 10, 20);
    expect(inv(fwd(17))).toBeCloseTo(17, 9);
  });
});

describe("niceTicks", () => {
  it("produces aligned ticks inside the domain", () => {
    const ticks = niceTicks(0.13, 0.97, 5);
    expect(ticks.length).toBeGreaterThanOrEqual(3);
    for (const t of ticks) {
      expect(t).toBeGreaterThanOrEqual(0.13 - 1e-9);
      expect(t).toBeLessThanOrEqual(0.97 + 1e-9);
    }
    // nice steps: differences should be uniform
    const diffs = ticks.slice(1).map((t, i) => t - ticks[i]);
    for (const d of diffs) expect(d).toBeCloseTo(diffs[0], 9);
  });

  it("handles zero-width and reversed domains", () => {
    expect(niceTicks(3, 3)).toEqual([3]);
    expect(niceTicks(9, 2)).toEqual(niceTicks(2, 9));
  });

  it("returns [] for non-finite input", () => {
    expect(niceTicks(Number.NaN, 1)).toEqual([]);
  });
});

describe("dateTicks", () => {
  const d = (s: string) => Date.parse(`${s}T00:00:00Z`);

  it("spans the domain with readable labels", () => {
    const ticks = dateTicks(d("2020-01-01"), d("2020-02-01"), 5);
    expect(ticks.length).toBeGreaterThanOrEqual(2);
    expect(ticks[0].label).toBe("2020-01-01");
  });

  it("switches to month labels for long spans", () => {
    const ticks = dateTicks(d("2020-01-01"), d("2024-01-01"), 5);
    for (const t of ticks) expect(t.label).toMatch(/^\d{4}-\d{2}$/);
  });

  it("equal bounds -> single tick; invalid -> []", () => {
    expect(dateTicks(1000, 1000)).toHaveLength(1);
    expect(dateTicks(5, 1)).toEqual([]);
  });
});

describe("decimateMinMax", () => {
  const mk = (ys: number[]): XY[] => ys.map((y, x) => ({ x, y }));

  it("keeps small series verbatim", () => {
    const pts = mk([1, 2, 3]);
    expect(decimateMinMax(pts, 10)).toEqual(pts);
  });

  it("preserves first, last, and the global extremes", () => {
    const ys = Array.from({ length: 1000 }, (_, i) =>
      i === 517 ? -50 : i === 220 ? 99 : i % 7,
    );
    const out = decimateMinMax(mk(ys), 50);
    expect(out[0].x).toBe(0);
    expect(out[out.length - 1].x).toBe(999);
    const outYs = out.map((p) => p.y);
    expect(outYs).toContain(-50);
    expect(outYs).toContain(99);
    expect(out.length).toBeLessThanOrEqual(2 * 50 + 2);
  });

  it("output x is monotonically non-decreasing", () => {
    const pts = mk(Array.from({ length: 500 }, (_, i) => Math.sin(i / 10) * i));
    const out = decimateMinMax(pts, 40);
    for (let i = 1; i < out.length; i += 1) {
      expect(out[i].x).toBeGreaterThanOrEqual(out[i - 1].x);
    }
  });

  it("empty input -> empty output", () => {
    expect(decimateMinMax([], 10)).toEqual([]);
  });
});

describe("path builders", () => {
  const x = linearScale(0, 10, 0, 100);
  const y = linearScale(0, 10, 100, 0);

  it("linePath emits M then L commands", () => {
    const d = linePath(
      [
        { x: 0, y: 0 },
        { x: 5, y: 5 },
        { x: 10, y: 10 },
      ],
      x,
      y,
    );
    expect(d.startsWith("M")).toBe(true);
    expect(d.match(/L/g)).toHaveLength(2);
    expect(linePath([], x, y)).toBe("");
  });

  it("areaPath closes against the baseline", () => {
    const d = areaPath(
      [
        { x: 0, y: 5 },
        { x: 10, y: 5 },
      ],
      x,
      y,
      0,
    );
    expect(d.endsWith("Z")).toBe(true);
    // baseline y=0 -> pixel 100 appears in the closing commands
    expect(d).toContain("100.00");
  });
});

describe("nearestIndex", () => {
  const pts: XY[] = [0, 10, 20, 30].map((x) => ({ x, y: 0 }));
  it("picks the closer neighbour", () => {
    expect(nearestIndex(pts, 12)).toBe(1);
    expect(nearestIndex(pts, 16)).toBe(2);
    expect(nearestIndex(pts, -50)).toBe(0);
    expect(nearestIndex(pts, 999)).toBe(3);
  });
  it("empty -> -1", () => {
    expect(nearestIndex([], 5)).toBe(-1);
  });
});

describe("extent + paddedExtent", () => {
  it("extent finds min/max; empty falls back", () => {
    expect(extent([3, 1, 7])).toEqual({ min: 1, max: 7 });
    expect(extent([])).toEqual({ min: 0, max: 1 });
  });
  it("paddedExtent pads and can anchor zero", () => {
    const p = paddedExtent([10, 20], 0.1);
    expect(p.min).toBeLessThan(10);
    expect(p.max).toBeGreaterThan(20);
    const z = paddedExtent([10, 20], 0.1, true);
    expect(z.min).toBe(0);
  });
});

describe("parseDay/isoDay", () => {
  it("round-trips UTC days", () => {
    expect(isoDay(parseDay("2024-02-29"))).toBe("2024-02-29");
  });
  it("rejects garbage", () => {
    expect(() => parseDay("not-a-date")).toThrow();
  });
});
