import { useEffect, useMemo, useState } from "react";
import type { FixtureIndex, ReceiptIndexEntry } from "../lib/types";
import { fetchBytes, fixturePath } from "../lib/fixtures";
import {
  extractHashFields,
  extractMeta,
  runClientChecks,
  sha256Hex,
  type ClientCheck,
} from "../lib/receiptVerify";
import { routeHref } from "../lib/router";
import { shortenHash } from "../lib/format";

interface ReceiptPageProps {
  entry: ReceiptIndexEntry | undefined;
  index: FixtureIndex;
}

export function ReceiptPage({ entry, index }: ReceiptPageProps) {
  const [receipt, setReceipt] = useState<unknown>(null);
  const [digest, setDigest] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!entry) return;
    let cancelled = false;
    setReceipt(null);
    setDigest(null);
    setError(null);
    fetchBytes(fixturePath(entry.file))
      .then(async (buf) => {
        const text = new TextDecoder().decode(buf);
        const parsed = JSON.parse(text) as unknown;
        const hex = await sha256Hex(buf);
        if (!cancelled) {
          setReceipt(parsed);
          setDigest(hex);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [entry]);

  const hashFields = useMemo(
    () => (receipt ? extractHashFields(receipt, index.hash_matches) : []),
    [receipt, index.hash_matches],
  );
  const checks: ClientCheck[] = useMemo(
    () => (receipt ? runClientChecks(receipt, hashFields, digest) : []),
    [receipt, hashFields, digest],
  );
  const meta = receipt ? extractMeta(receipt) : null;

  if (!entry) {
    return (
      <main className="page">
        <p>
          Unknown receipt.{" "}
          <a href={routeHref({ name: "receipts" })}>Back to receipts</a>
        </p>
      </main>
    );
  }

  return (
    <main className="page" data-testid="receipt-page">
      <nav className="breadcrumb">
        <a href={routeHref({ name: "receipts" })}>Receipts</a> / {entry.id}
      </nav>
      <header className="page-head">
        <h2 className="mono">{entry.id}</h2>
        <div className="badges">
          {meta?.schema && <span className="badge">{meta.schema}</span>}
          {meta?.evidence_level && (
            <span className="badge">{meta.evidence_level}</span>
          )}
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {!receipt && !error && <p className="muted">loading…</p>}

      {receipt !== null && meta !== null && (
        <>
          <section className="panel" data-testid="meta-panel">
            <h3>Declared metadata</h3>
            <dl className="meta-grid">
              <dt>schema</dt>
              <dd>{meta.schema ?? "—"}</dd>
              <dt>evidence_level</dt>
              <dd>{meta.evidence_level ?? "—"}</dd>
              <dt>created</dt>
              <dd>{meta.created_at ?? "—"}</dd>
              <dt>research_only</dt>
              <dd>{String(meta.research_only)}</dd>
              <dt>live_pnl_claim</dt>
              <dd>{String(meta.live_pnl_claim)}</dd>
              <dt>data_source</dt>
              <dd>{meta.data_source ?? "—"}</dd>
            </dl>
            {meta.disclaimer && (
              <blockquote className="disclaimer">{meta.disclaimer}</blockquote>
            )}
          </section>

          <section className="panel" data-testid="checks-panel">
            <h3>Client-side checks</h3>
            <p className="muted">
              Informational only — computed in-browser from the fixture copy.
              Authoritative, fail-closed verification:{" "}
              <code>dipcatcher verify-research</code>.
            </p>
            <ul className="check-list">
              {checks.map((c) => (
                <li key={c.id} className={`check check-${c.status}`}>
                  <span className="check-status">{statusGlyph(c.status)}</span>
                  <span className="check-label">{c.label}</span>
                  <span className="check-detail mono">{c.detail}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="panel" data-testid="hash-panel">
            <h3>Digest fields ({hashFields.length})</h3>
            {hashFields.length === 0 ? (
              <p className="muted">No sha256-format fields in this receipt.</p>
            ) : (
              <table className="list-table hash-table">
                <thead>
                  <tr>
                    <th>field</th>
                    <th>digest</th>
                    <th>checkout match</th>
                  </tr>
                </thead>
                <tbody>
                  {hashFields.map((h) => (
                    <tr key={h.key}>
                      <td className="mono">{h.key}</td>
                      <td className="mono" title={h.value}>
                        {shortenHash(h.value)}
                      </td>
                      <td>
                        {h.match ? (
                          <span className="badge pass" title={h.match}>
                            {h.match}
                          </span>
                        ) : (
                          <span className="badge">unresolved</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="muted">
              "checkout match" = the digest equals raw file bytes (or the
              canonical-JSON re-encoding) of a committed file at
              fixture-generation time. Unresolved ≠ invalid — sealed hashes may
              point at inputs outside this checkout or at prior revisions.
            </p>
          </section>

          <section className="panel">
            <details>
              <summary>Raw receipt JSON</summary>
              <pre className="raw-json" data-testid="raw-json">
                {JSON.stringify(receipt, null, 1)}
              </pre>
            </details>
          </section>
        </>
      )}
    </main>
  );
}

function statusGlyph(s: ClientCheck["status"]): string {
  return s === "pass" ? "✓" : s === "fail" ? "✗" : "ℹ";
}
