import { describe, expect, it } from "vitest";
import { parseHash, routeHref } from "./router";

describe("parseHash", () => {
  it("empty/root hashes -> overview", () => {
    expect(parseHash("")).toEqual({ name: "overview" });
    expect(parseHash("#")).toEqual({ name: "overview" });
    expect(parseHash("#/")).toEqual({ name: "overview" });
  });

  it("parses strategy + receipt routes", () => {
    expect(parseHash("#/strategy/carry")).toEqual({
      name: "strategy",
      id: "carry",
    });
    expect(parseHash("#/receipts")).toEqual({ name: "receipts" });
    expect(parseHash("#/receipt/dip_bench_crypto_1d_20260925")).toEqual({
      name: "receipt",
      id: "dip_bench_crypto_1d_20260925",
    });
  });

  it("decodes uri components and ignores trailing path", () => {
    expect(parseHash("#/strategy/a%20b")).toEqual({
      name: "strategy",
      id: "a b",
    });
    expect(parseHash("#/strategy/carry/extra")).toEqual({
      name: "strategy",
      id: "carry",
    });
    expect(parseHash("#/unknown/route")).toEqual({ name: "overview" });
  });

  it("falls back to overview on malformed uri components", () => {
    // decodeURIComponent throws URIError on truncated/invalid escapes —
    // a bad hash must route to overview, not crash the app.
    expect(parseHash("#/receipt/%")).toEqual({ name: "overview" });
    expect(parseHash("#/receipt/%E0%A4%A")).toEqual({ name: "overview" });
    expect(parseHash("#/strategy/%zz")).toEqual({ name: "overview" });
    expect(parseHash("#/receipt/valid_id")).toEqual({
      name: "receipt",
      id: "valid_id",
    });
  });
});

describe("routeHref", () => {
  it("builds hash hrefs and encodes ids", () => {
    expect(routeHref({ name: "overview" })).toBe("#/");
    expect(routeHref({ name: "receipts" })).toBe("#/receipts");
    expect(routeHref({ name: "strategy", id: "a b" })).toBe(
      "#/strategy/a%20b",
    );
    expect(routeHref({ name: "receipt", id: "r1" })).toBe("#/receipt/r1");
  });

  it("round-trips through parseHash", () => {
    for (const route of [
      { name: "overview" },
      { name: "receipts" },
      { name: "strategy", id: "amix_sleeve_fade" },
      { name: "receipt", id: "incumbent_bench_qlib" },
    ] as const) {
      expect(parseHash(routeHref(route))).toEqual(route);
    }
  });
});
