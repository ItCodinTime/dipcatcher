import { useMemo } from "react";
import type { EquityPoint, FixtureIndex, StrategyIndexEntry } from "../lib/types";
import { decodeEquity, fetchJson, fixturePath, validateStrategy } from "../lib/fixtures";
import { drawdownSeries, summarizeDrawdown } from "../lib/drawdown";
import { parseDay } from "../lib/chart";
import { fmtPct, fmtUSD } from "../lib/format";
import { routeHref } from "../lib/router";
import { flattenDict } from "../lib/table";
import { useAsync } from "../lib/useJson";
import { LineChart } from "./LineChart";
import { StatsTable } from "./StatsTable";

interface StrategyPageProps {
  entry: StrategyIndexEntry | undefined;
  index: FixtureIndex;
}

export function StrategyPage({ entry, index }: StrategyPageProps) {
  const stats = useAsync(
    entry ? entry.stats_file : null,
    entry
      ? async () =>
          validateStrategy(await fetchJson(fixturePath(entry.stats_file)))
      : null,
  );
  const equity = useAsync(
    entry?.equity_file ?? null,
    entry?.equity_file
      ? async () =>
          decodeEquity(await fetchJson(fixturePath(entry.equity_file!)))
      : null,
  );

  const derived = useMemo(() => {
    if (!equity.data || equity.data.length === 0) return null;
    const pts: EquityPoint[] = equity.data;
    const dd = drawdownSeries(pts);
    return {
      nav: pts.map((p) => ({ x: parseDay(p.date), y: p.nav })),
      dd: dd.map((p) => ({ x: parseDay(p.date), y: p.drawdown })),
      ddSummary: summarizeDrawdown(dd),
      first: pts[0],
      last: pts[pts.length - 1],
      rows: pts.length,
    };
  }, [equity.data]);

  if (!entry) {
    return (
      <main className="page">
        <p>
          Unknown strategy.{" "}
          <a href={routeHref({ name: "overview" })}>Back to overview</a>
        </p>
      </main>
    );
  }

  const linkedReceipt = stats.data?.provenance.receipt
    ? index.receipts.find(
        (r) => r.file === stats.data!.provenance.receipt,
      )
    : undefined;

  return (
    <main className="page" data-testid="strategy-page">
      <nav className="breadcrumb">
        <a href={routeHref({ name: "overview" })}>Strategies</a> / {entry.id}
      </nav>
      <header className="page-head">
        <h2>{stats.data?.name ?? entry.name}</h2>
        <div className="badges">
          <span className={`badge kind-${entry.kind}`}>{entry.kind}</span>
          <span
            className={
              entry.data_source.toUpperCase() === "SYNTHETIC"
                ? "badge synthetic"
                : "badge"
            }
          >
            {entry.data_source}
          </span>
          <span className="badge research">research_only</span>
        </div>
      </header>

      {stats.error && <p className="error">stats: {stats.error}</p>}
      {stats.data?.provenance.note && (
        <p className="muted provenance">{stats.data.provenance.note}</p>
      )}

      {entry.has_equity && (
        <section className="panel">
          <h3>Simulated equity (NAV)</h3>
          {equity.loading && <p className="muted">loading…</p>}
          {equity.error && <p className="error">equity: {equity.error}</p>}
          {derived && (
            <>
              <div className="chip-row" data-testid="equity-summary">
                <span className="chip">{derived.rows} daily bars</span>
                <span className="chip">
                  {derived.first.date} → {derived.last.date}
                </span>
                <span className="chip">
                  final NAV {fmtUSD(derived.last.nav)}
                </span>
                <span className="chip">
                  max drawdown {fmtPct(-derived.ddSummary.maxDrawdown)}
                </span>
                <span className="chip">
                  underwater {fmtPct(derived.ddSummary.underwaterFraction, 0)}{" "}
                  of days
                </span>
              </div>
              <LineChart
                points={derived.nav}
                yFormat={(v) => fmtUSD(v)}
                color="#2563eb"
                anchorZero={false}
                ariaLabel={`Simulated NAV for ${entry.name}`}
              />
              <h3>Drawdown</h3>
              <LineChart
                points={derived.dd}
                yFormat={(v) => fmtPct(v)}
                color="#dc2626"
                fillBaseline={0}
                anchorZero
                ariaLabel={`Drawdown for ${entry.name}`}
                height={200}
              />
            </>
          )}
        </section>
      )}

      {!entry.has_equity && (
        <p className="muted" data-testid="no-equity-note">
          No equity series exported for this entry — recorded segment stats
          only.
        </p>
      )}

      <section className="panel">
        <h3>Per-segment stats (as recorded)</h3>
        {stats.loading && <p className="muted">loading…</p>}
        {stats.data && <StatsTable segments={stats.data.segments} />}
      </section>

      {stats.data && flattenDict(stats.data.extras).length > 0 && (
        <section className="panel">
          <h3>Full-path extras</h3>
          <table className="list-table" data-testid="extras-table">
            <tbody>
              {flattenDict(stats.data.extras).map((row) => (
                <tr key={row.key}>
                  <th scope="row" className="mono">
                    {row.key}
                  </th>
                  <td>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="panel">
        <h3>Provenance</h3>
        <ul className="provenance-list">
          {stats.data?.provenance.source_files?.map((f) => (
            <li key={f} className="mono">
              {f}
            </li>
          ))}
          {linkedReceipt && (
            <li>
              receipt:{" "}
              <a href={routeHref({ name: "receipt", id: linkedReceipt.id })}>
                {linkedReceipt.id}
              </a>
            </li>
          )}
          {!linkedReceipt && stats.data?.provenance.receipt && (
            <li className="mono">{stats.data.provenance.receipt}</li>
          )}
          {stats.data?.provenance.pointer && (
            <li className="mono">pointer: {stats.data.provenance.pointer}</li>
          )}
        </ul>
      </section>
    </main>
  );
}
