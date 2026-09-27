import type { FixtureIndex } from "../lib/types";
import { routeHref } from "../lib/router";

export function ReceiptsPage({ index }: { index: FixtureIndex }) {
  return (
    <main className="page" data-testid="receipts-page">
      <header className="page-head">
        <h2>Sealed research receipts</h2>
        <p className="muted">
          Verbatim copies of files under <code>receipts/</code>. Digest fields
          are listed on each receipt's verification panel; the authoritative
          check remains <code>dipcatcher verify-research</code> in-repo.
        </p>
      </header>
      <table className="list-table" data-testid="receipt-table">
        <thead>
          <tr>
            <th>receipt</th>
            <th>schema</th>
            <th>evidence level</th>
            <th>created</th>
            <th>research_only</th>
            <th>live_pnl_claim</th>
            <th>digests resolved</th>
          </tr>
        </thead>
        <tbody>
          {index.receipts.map((r) => (
            <tr key={r.id}>
              <td>
                <a href={routeHref({ name: "receipt", id: r.id })}>{r.id}</a>
              </td>
              <td className="mono">{r.schema ?? "—"}</td>
              <td className="muted">{r.evidence_level ?? "—"}</td>
              <td className="muted">{(r.created_at ?? "").slice(0, 10) || "—"}</td>
              <td>
                <FlagBadge value={r.research_only} expect={true} />
              </td>
              <td>
                <FlagBadge value={r.live_pnl_claim} expect={false} />
              </td>
              <td>
                {r.n_hash_matches}/{r.n_hashes}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}

function FlagBadge({
  value,
  expect,
}: {
  value: boolean | null;
  expect: boolean;
}) {
  if (value === null) return <span className="badge">absent</span>;
  const ok = value === expect;
  return (
    <span className={`badge ${ok ? "pass" : "fail"}`}>{String(value)}</span>
  );
}
