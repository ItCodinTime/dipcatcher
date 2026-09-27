/** Receipt inspection: hash-field extraction + client-side checks.
 *
 * The authoritative verifier is `dipcatcher verify-research`; this module only
 * surfaces what a static page can honestly compute: which sha256-format
 * fields exist, whether any resolve to files committed in this checkout (via
 * the export-time `hash_matches` table), and whether the receipt's declared
 * honesty flags are intact.
 */

export interface HashField {
  /** Dotted path, e.g. `script_sha256` or `input_hashes.btcusdt_1d.parquet`. */
  key: string;
  value: string;
  /** Repo path from `index.hash_matches`, or null when unresolved. */
  match: string | null;
  kind: "field" | "table-entry";
}

export interface ReceiptMeta {
  schema: string | null;
  evidence_level: string | null;
  research_only: boolean | null;
  live_pnl_claim: boolean | null;
  created_at: string | null;
  disclaimer: string | null;
  data_source: string | null;
}

const SHA256_RE = /^[0-9a-f]{64}$/;

function isDigest(v: unknown): v is string {
  return typeof v === "string" && SHA256_RE.test(v);
}

/** Keys whose values are dicts of name -> digest (not digests themselves). */
function isHashTableKey(key: string): boolean {
  const k = key.toLowerCase();
  return k.endsWith("_hashes") || (k.includes("hash") && !isDigestKey(k));
}

function isDigestKey(key: string): boolean {
  const k = key.toLowerCase();
  return k === "sha256" || k.endsWith("_sha256") || k.endsWith("_sha");
}

/**
 * Depth-first walk collecting digest fields. Digest-valued dict entries are
 * emitted as `parentKey.entryName`; non-digest values under `*_hashes` tables
 * are ignored (they are names, not hashes).
 */
export function extractHashFields(
  receipt: unknown,
  matches: Record<string, string> = {},
): HashField[] {
  const out: HashField[] = [];

  function walk(node: unknown, path: string, insideHashTable: boolean): void {
    if (node === null || typeof node !== "object") return;
    if (Array.isArray(node)) {
      node.forEach((item, i) => walk(item, `${path}[${i}]`, false));
      return;
    }
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      const childPath = path ? `${path}.${key}` : key;
      const hashyKey = isDigestKey(key) || isHashTableKey(key);
      if (isDigest(value)) {
        // Conservative: only label 64-hex strings as digests when the key is
        // hash-ish (script_sha256) or they sit inside a hash table
        // (input_hashes.<file>) — mirrors the export script's collector.
        if (insideHashTable || hashyKey) {
          out.push({
            key: childPath,
            value,
            match: matches[value] ?? null,
            kind: insideHashTable ? "table-entry" : "field",
          });
        }
        continue;
      }
      walk(value, childPath, hashyKey);
    }
  }

  walk(receipt, "", false);
  return out;
}

export function extractMeta(receipt: unknown): ReceiptMeta {
  const r = (receipt ?? {}) as Record<string, unknown>;
  const str = (v: unknown): string | null =>
    typeof v === "string" && v.length > 0 ? v : null;
  const bool = (v: unknown): boolean | null =>
    typeof v === "boolean" ? v : null;
  return {
    schema: str(r.schema),
    evidence_level: str(r.evidence_level),
    research_only: bool(r.research_only),
    live_pnl_claim: bool(r.live_pnl_claim),
    created_at: str(r.created_at) ?? str(r.generated_at),
    disclaimer: str(r.disclaimer),
    data_source: str(r.data_source) ?? str(r.data),
  };
}

export type CheckStatus = "pass" | "fail" | "info";

export interface ClientCheck {
  id: string;
  label: string;
  status: CheckStatus;
  detail: string;
}

/**
 * Informational checks a static page can perform. This is NOT receipt
 * verification — that stays fail-closed in `dipcatcher verify-research`; the
 * UI labels this block accordingly.
 */
export function runClientChecks(
  receipt: unknown,
  hashFields: HashField[],
  fileDigest: string | null,
): ClientCheck[] {
  const meta = extractMeta(receipt);
  const checks: ClientCheck[] = [];

  checks.push({
    id: "file-digest",
    label: "Fetched-file SHA-256",
    status: "info",
    detail: fileDigest ?? "unavailable (no WebCrypto)",
  });

  checks.push({
    id: "research-only",
    label: "research_only flag",
    status:
      meta.research_only === true
        ? "pass"
        : meta.research_only === null
          ? "info"
          : "fail",
    detail:
      meta.research_only === null
        ? "flag absent from receipt"
        : `research_only = ${meta.research_only}`,
  });

  checks.push({
    id: "live-pnl",
    label: "live_pnl_claim flag",
    status:
      meta.live_pnl_claim === false
        ? "pass"
        : meta.live_pnl_claim === null
          ? "info"
          : "fail",
    detail:
      meta.live_pnl_claim === null
        ? "flag absent from receipt"
        : `live_pnl_claim = ${meta.live_pnl_claim}`,
  });

  const nMatches = hashFields.filter((h) => h.match !== null).length;
  checks.push({
    id: "hash-fields",
    label: "sha256 fields resolved to checkout",
    status: hashFields.length === 0 ? "info" : nMatches > 0 ? "pass" : "info",
    detail:
      hashFields.length === 0
        ? "no sha256-format fields in receipt"
        : `${nMatches}/${hashFields.length} digests match a committed file`,
  });

  checks.push({
    id: "schema",
    label: "schema identifier",
    status: meta.schema ? "info" : "fail",
    detail: meta.schema ?? "missing schema field",
  });

  return checks;
}

/** Browser SHA-256 of the exact fetched bytes; null when unsupported. */
export async function sha256Hex(bytes: ArrayBuffer): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = await subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
