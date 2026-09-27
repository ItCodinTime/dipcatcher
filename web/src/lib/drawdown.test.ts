import { describe, expect, it } from "vitest";
import { drawdownSeries, simpleReturns, summarizeDrawdown } from "./drawdown";

const nav = (values: number[]) =>
  values.map((v, i) => ({ date: `2024-01-${String(i + 1).padStart(2, "0")}`, nav: v }));

describe("drawdownSeries", () => {
  it("is zero while the path makes new highs", () => {
    const dd = drawdownSeries(nav([100, 110, 120]));
    expect(dd.map((d) => d.drawdown)).toEqual([0, 0, 0]);
    expect(dd[2].peak).toBe(120);
  });

  it("measures the fractional decline from the running peak", () => {
    const dd = drawdownSeries(nav([100, 150, 120, 160]));
    expect(dd[2].drawdown).toBeCloseTo(1 - 120 / 150, 9);
    expect(dd[3].drawdown).toBe(0);
  });

  it("handles a monotonic decline", () => {
    const dd = drawdownSeries(nav([100, 90, 80]));
    expect(dd[2].drawdown).toBeCloseTo(0.2, 9);
  });

  it("empty -> empty", () => {
    expect(drawdownSeries([])).toEqual([]);
  });
});

describe("summarizeDrawdown", () => {
  it("reports max drawdown, its date, and underwater fraction", () => {
    const dd = drawdownSeries(nav([100, 200, 100, 150, 200]));
    const s = summarizeDrawdown(dd);
    expect(s.maxDrawdown).toBeCloseTo(0.5, 9);
    expect(s.maxDrawdownDate).toBe("2024-01-03");
    expect(s.currentDrawdown).toBe(0);
    expect(s.underwaterFraction).toBeCloseTo(2 / 5, 9);
  });

  it("empty series -> zeros", () => {
    const s = summarizeDrawdown([]);
    expect(s.maxDrawdown).toBe(0);
    expect(s.maxDrawdownDate).toBeNull();
  });
});

describe("simpleReturns", () => {
  it("computes period returns and skips zero previous nav", () => {
    const r = simpleReturns(nav([100, 110, 0, 55]));
    expect(r).toHaveLength(2);
    expect(r[0].ret).toBeCloseTo(0.1, 9);
    // nav 0 -> 55 has no finite previous, so last point is 0/110
    expect(r[1].ret).toBeCloseTo(-1, 9);
  });
});
