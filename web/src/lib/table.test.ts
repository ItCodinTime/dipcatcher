import { describe, expect, it } from "vitest";
import { flattenDict, segmentMetricRows } from "./table";

describe("segmentMetricRows", () => {
  it("orders known metrics and appends unknown alphabetically", () => {
    const rows = segmentMetricRows({
      dev: { sharpe: 1, n: 10, zeta: 1, alpha: 2 },
      holdout: { max_drawdown: -0.1, net_return: 0.2, n: 8 },
    });
    expect(rows).toEqual([
      "n",
      "net_return",
      "sharpe",
      "max_drawdown",
      "alpha",
      "zeta",
    ]);
  });

  it("empty -> empty", () => {
    expect(segmentMetricRows({})).toEqual([]);
  });
});

describe("flattenDict", () => {
  it("flattens nested dicts into dotted keys", () => {
    const rows = flattenDict({
      full_path_costs: { commission: 10.5, spread: 2 },
      full_path_liquidations: 0,
    });
    expect(rows).toEqual([
      { key: "full_path_costs.commission", value: "10.5000" },
      { key: "full_path_costs.spread", value: "2" },
      { key: "full_path_liquidations", value: "0" },
    ]);
  });

  it("handles scalars, nulls, arrays", () => {
    const rows = flattenDict({ a: true, b: null, c: [1, 2] });
    expect(rows).toContainEqual({ key: "a", value: "true" });
    expect(rows).toContainEqual({ key: "b", value: "—" });
    expect(rows).toContainEqual({ key: "c.[0]", value: "1" });
  });

  it("respects the depth cap and non-objects", () => {
    expect(flattenDict(42)).toEqual([]);
    expect(flattenDict({ a: { b: { c: { d: { e: { f: 1 } } } } } })).toEqual(
      [],
    );
  });
});
