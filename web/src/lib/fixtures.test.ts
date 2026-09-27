import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { decodeEquity, validateIndex } from "./fixtures";

const FIXTURES = join(import.meta.dirname, "..", "..", "public", "fixtures");

describe("validateIndex", () => {
  it("accepts the real generated index", () => {
    const raw = JSON.parse(
      readFileSync(join(FIXTURES, "index.json"), "utf8"),
    ) as unknown;
    const idx = validateIndex(raw);
    expect(idx.strategies.length).toBeGreaterThanOrEqual(9);
    expect(idx.receipts).toHaveLength(6);
    expect(idx.honesty.research_only).toBe(true);
    const carry = idx.strategies.find((s) => s.id === "carry")!;
    expect(carry.has_equity).toBe(true);
    expect(carry.equity_file).toBe("equity/carry.json");
  });

  it("rejects non-objects and missing arrays", () => {
    expect(() => validateIndex(null)).toThrow();
    expect(() => validateIndex({ strategies: [] })).toThrow(/receipts/);
    expect(() =>
      validateIndex({ strategies: "nope", receipts: [] }),
    ).toThrow();
    expect(() =>
      validateIndex({ strategies: [{ name: "x" }], receipts: [] }),
    ).toThrow(/id/);
  });

  it("tolerates absent optional fields", () => {
    const idx = validateIndex({
      strategies: [{ id: "a", name: "A", stats_file: "s/a.json" }],
      receipts: [{ id: "r", file: "receipts/r.json" }],
    });
    expect(idx.strategies[0].kind).toBe("unknown");
    expect(idx.strategies[0].has_equity).toBe(false);
    expect(idx.receipts[0].schema).toBeNull();
    expect(idx.repo_revision).toBeNull();
  });
});

describe("decodeEquity", () => {
  it("decodes the real carry fixture into sorted points", () => {
    const raw = JSON.parse(
      readFileSync(join(FIXTURES, "equity", "carry.json"), "utf8"),
    ) as unknown;
    const pts = decodeEquity(raw);
    expect(pts.length).toBeGreaterThan(2000);
    expect(pts[0].date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(pts[0].nav).toBeGreaterThan(0);
    for (let i = 1; i < pts.length; i += 1) {
      expect(pts[i].date >= pts[i - 1].date).toBe(true);
      expect(Number.isFinite(pts[i].nav)).toBe(true);
    }
  });

  it("rejects malformed payloads", () => {
    expect(() => decodeEquity(null)).toThrow();
    expect(() => decodeEquity({})).toThrow(/points/);
    expect(() => decodeEquity({ points: [[1, 2]] })).toThrow(/malformed/);
  });
});
