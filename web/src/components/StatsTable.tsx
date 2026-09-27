import type { SegmentStats } from "../lib/types";
import { segmentMetricRows } from "../lib/table";
import { fmtMetric } from "../lib/format";

interface StatsTableProps {
  segments: Record<string, SegmentStats>;
  caption?: string;
}

/** Metrics-as-rows x segments-as-columns table of stored stats. */
export function StatsTable({ segments, caption }: StatsTableProps) {
  const names = Object.keys(segments);
  if (names.length === 0) {
    return <p className="muted">No segment stats recorded.</p>;
  }
  const rows = segmentMetricRows(segments);
  return (
    <table className="stats-table" data-testid="stats-table">
      {caption && <caption>{caption}</caption>}
      <thead>
        <tr>
          <th>metric</th>
          {names.map((n) => (
            <th key={n}>{n}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((metric) => (
          <tr key={metric}>
            <th scope="row">{metric}</th>
            {names.map((seg) => (
              <td key={seg}>
                {metric in segments[seg]
                  ? fmtMetric(metric, segments[seg][metric])
                  : "—"}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
