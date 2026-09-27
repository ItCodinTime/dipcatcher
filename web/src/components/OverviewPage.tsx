import type { FixtureIndex } from "../lib/types";
import { routeHref } from "../lib/router";

/** Strategy list + receipt summary. Headline = index metadata only. */
export function OverviewPage({ index }: { index: FixtureIndex }) {
  return (
    <main className="page" data-testid="overview-page">
      <section className="panel">
        <h2>Strategies</h2>
        <p className="muted">
          Stat blobs and simulated equity paths exported from committed
          artifacts and sealed receipts. "Segments" are the regimes recorded
          by each source (e.g. development / holdout).
        </p>
        <table className="list-table" data-testid="strategy-table">
          <thead>
            <tr>
              <th>strategy</th>
              <th>kind</th>
              <th>data</th>
              <th>segments</th>
              <th>equity</th>
            </tr>
          </thead>
          <tbody>
            {index.strategies.map((s) => (
              <tr key={s.id}>
                <td>
                  <a href={routeHref({ name: "strategy", id: s.id })}>
                    {s.name}
                  </a>
                </td>
                <td>
                  <span className={`badge kind-${s.kind}`}>{s.kind}</span>
                </td>
                <td>
                  <span
                    className={
                      s.data_source.toUpperCase() === "SYNTHETIC"
                        ? "badge synthetic"
                        : "badge"
                    }
                  >
                    {s.data_source}
                  </span>
                </td>
                <td className="muted">{s.segments.join(", ") || "—"}</td>
                <td>{s.has_equity ? "series" : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>
          Receipts{" "}
          <a href={routeHref({ name: "receipts" })} className="subtle-link">
            view all →
          </a>
        </h2>
        <table className="list-table" data-testid="receipt-summary-table">
          <thead>
            <tr>
              <th>receipt</th>
              <th>schema</th>
              <th>evidence level</th>
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
                <td>
                  {r.n_hash_matches}/{r.n_hashes}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <footer className="muted footer-meta">
        fixtures generated {index.generated_at || "unknown"} · repo{" "}
        {index.repo_revision ? index.repo_revision.slice(0, 12) : "unknown"}
      </footer>
    </main>
  );
}
