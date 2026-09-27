import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  extractHashFields,
  extractMeta,
  runClientChecks,
  sha256Hex,
} from "./receiptVerify";

const FIXTURES = join(import.meta.dirname, "..", "..", "public", "fixtures");
const REAL_RECEIPT = JSON.parse(
  readFileSync(
    join(FIXTURES, "receipts", "adaptive_mix_20asset_1d_20260922.json"),
    "utf8",
  ),
) as Record<string, unknown>;
const REAL_INDEX = JSON.parse(
  readFileSync(join(FIXTURES, "index.json"), "utf8"),
) as { hash_matches: Record<string, string> };

const h = (c: string) => c.repeat(64);

describe("extractHashFields", () => {
  it("collects top-level digest fields", () => {
    const r = { script_sha256: h("a"), schema: "x.v1" };
    const fields = extractHashFields(r);
    expect(fields).toHaveLength(1);
    expect(fields[0]).toMatchObject({
      key: "script_sha256",
      value: h("a"),
      kind: "field",
      match: null,
    });
  });

  it("collects entries under hash-table keys with dotted names", () => {
    const r = {
      input_hashes: { "btc.parquet": h("b"), "eth.parquet": h("c") },
      nested: { inputs_sha256: { "x.parquet": h("d") } },
    };
    const keys = extractHashFields(r).map((f) => f.key);
    expect(keys).toEqual(
      expect.arrayContaining([
        "input_hashes.btc.parquet",
        "input_hashes.eth.parquet",
        "nested.inputs_sha256.x.parquet",
      ]),
    );
    const kinds = extractHashFields(r).map((f) => f.kind);
    expect(new Set(kinds)).toEqual(new Set(["table-entry"]));
  });

  it("ignores non-hex strings and non-digest dict members", () => {
    const r = {
      script_sha256: "not-a-hash",
      input_hashes: { name: "zzz" },
      note: h("e"),
    };
    expect(extractHashFields(r)).toHaveLength(0);
  });

  it("resolves matches via the index hash_matches table", () => {
    const r = { script_sha256: h("a") };
    const fields = extractHashFields(r, { [h("a")]: "scripts/x.py" });
    expect(fields[0].match).toBe("scripts/x.py");
  });

  it("handles the real adaptive-mix receipt (6 fields + 40 input hashes)", () => {
    const fields = extractHashFields(REAL_RECEIPT, REAL_INDEX.hash_matches);
    expect(fields).toHaveLength(46);
    const fieldKeys = fields
      .filter((f) => f.kind === "field")
      .map((f) => f.key);
    expect(fieldKeys).toEqual(
      expect.arrayContaining([
        "script_sha256",
        "manifest_sha256",
        "config_sha256",
        "metrics_sha256",
        "engine_sha256",
        "allocator_sha256",
      ]),
    );
    // the export-time scan matched metrics_sha256 to a committed file
    const metrics = fields.find((f) => f.key === "metrics_sha256");
    expect(metrics?.match).toBe("src/quant_fund/metrics/returns.py");
  });
});

describe("extractMeta", () => {
  it("reads declared fields and falls back for created_at", () => {
    const meta = extractMeta({
      schema: "s.v1",
      generated_at: "2026-01-01T00:00:00Z",
      research_only: true,
      live_pnl_claim: false,
      disclaimer: "text",
    });
    expect(meta.schema).toBe("s.v1");
    expect(meta.created_at).toBe("2026-01-01T00:00:00Z");
    expect(meta.research_only).toBe(true);
  });

  it("tolerates a bare object", () => {
    const meta = extractMeta({});
    expect(meta.schema).toBeNull();
    expect(meta.research_only).toBeNull();
  });
});

describe("runClientChecks", () => {
  it("passes honesty flags on a well-formed receipt", () => {
    const checks = runClientChecks(REAL_RECEIPT, [], h("f"));
    const byId = Object.fromEntries(checks.map((c) => [c.id, c]));
    expect(byId["research-only"].status).toBe("pass");
    expect(byId["live-pnl"].status).toBe("pass");
    expect(byId["file-digest"].detail).toBe(h("f"));
    expect(byId["hash-fields"].detail).toContain(
      "no sha256-format fields",
    );
  });

  it("fails loudly on a live-PnL claim", () => {
    const checks = runClientChecks(
      { research_only: false, live_pnl_claim: true, schema: "x" },
      [],
      null,
    );
    const byId = Object.fromEntries(checks.map((c) => [c.id, c]));
    expect(byId["research-only"].status).toBe("fail");
    expect(byId["live-pnl"].status).toBe("fail");
  });

  it("reports digest resolution counts", () => {
    const hashFields = [
      { key: "a_sha256", value: h("1"), match: "src/x.py", kind: "field" as const },
      { key: "b_sha256", value: h("2"), match: null, kind: "field" as const },
    ];
    const checks = runClientChecks({ schema: "x" }, hashFields, null);
    const c = checks.find((c) => c.id === "hash-fields")!;
    expect(c.detail).toBe("1/2 digests match a committed file");
    expect(c.status).toBe("pass");
  });
});

describe("sha256Hex", () => {
  it("hashes bytes via WebCrypto", async () => {
    const bytes = new TextEncoder().encode("abc").buffer as ArrayBuffer;
    const hex = await sha256Hex(bytes);
    // well-known sha256("abc")
    expect(hex).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});
