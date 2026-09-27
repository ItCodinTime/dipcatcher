import { useMemo, useState, type MouseEvent as ReactMouseEvent } from "react";
import {
  areaPath,
  dateTicks,
  decimateMinMax,
  invertScale,
  isoDay,
  linePath,
  linearScale,
  nearestIndex,
  niceTicks,
  paddedExtent,
} from "../lib/chart";
import type { XY } from "../lib/chart";

const W = 720;
const MARGIN = { top: 12, right: 18, bottom: 26, left: 68 };

interface LineChartProps {
  /** x = epoch-ms, y = value. Must be sorted by x. */
  points: XY[];
  height?: number;
  color?: string;
  /** Format a y value for ticks + tooltip. */
  yFormat?: (v: number) => string;
  /** Format an x value (epoch-ms) for the tooltip. */
  xFormat?: (ms: number) => string;
  /** When set, fill the area between the line and this domain y value. */
  fillBaseline?: number | null;
  /** Force y domain to include 0 when all values are >= 0. */
  anchorZero?: boolean;
  ariaLabel: string;
}

/**
 * Hand-rolled responsive SVG line chart with gridlines, nice ticks and a
 * nearest-point hover readout. Rendering uses min-max decimation while hover
 * resolves against the full series.
 */
export function LineChart({
  points,
  height = 260,
  color = "#3b82f6",
  yFormat = (v) => v.toPrecision(4),
  xFormat = isoDay,
  fillBaseline = null,
  anchorZero = false,
  ariaLabel,
}: LineChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const H = height;
  const innerW = W - MARGIN.left - MARGIN.right;
  const innerH = H - MARGIN.top - MARGIN.bottom;

  const { renderPts, yTicks, xTicks, x, y, xInv } = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xMin = xs.length ? Math.min(...xs) : 0;
    const xMax = xs.length ? Math.max(...xs) : 1;
    const yd = paddedExtent(
      fillBaseline !== null ? [...ys, fillBaseline] : ys,
      0.06,
      anchorZero,
    );
    const xs_ = linearScale(xMin, xMax, MARGIN.left, MARGIN.left + innerW);
    const ys_ = linearScale(yd.min, yd.max, MARGIN.top + innerH, MARGIN.top);
    return {
      renderPts: decimateMinMax(points, 360),
        yTicks: niceTicks(yd.min, yd.max, 5),
        xTicks: dateTicks(xMin, xMax, 6),
        x: xs_,
        y: ys_,
        xInv: invertScale(MARGIN.left, MARGIN.left + innerW, xMin, xMax),
    };
  }, [points, fillBaseline, anchorZero, innerW, innerH]);

  const onMove = (e: ReactMouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * W;
    const idx = nearestIndex(points, xInv(svgX));
    setHover(idx >= 0 ? idx : null);
  };

  const hoverPt = hover !== null ? points[hover] : null;

  return (
    <div className="chart" data-testid="line-chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* gridlines + y ticks */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={MARGIN.left}
              x2={W - MARGIN.right}
              y1={y(t)}
              y2={y(t)}
              className="grid"
            />
            <text
              x={MARGIN.left - 8}
              y={y(t) + 4}
              className="tick"
              textAnchor="end"
            >
              {yFormat(t)}
            </text>
          </g>
        ))}
        {/* x ticks */}
        {xTicks.map((t) => (
          <text
            key={t.ms}
            x={x(t.ms)}
            y={H - MARGIN.bottom + 18}
            className="tick"
            textAnchor="middle"
          >
            {t.label}
          </text>
        ))}
        {/* baseline */}
        {fillBaseline !== null && (
          <line
            x1={MARGIN.left}
            x2={W - MARGIN.right}
            y1={y(fillBaseline)}
            y2={y(fillBaseline)}
            className="baseline"
          />
        )}
        {fillBaseline !== null && (
          <path
            d={areaPath(renderPts, x, y, fillBaseline)}
            className="area"
            fill={color}
          />
        )}
        <path d={linePath(renderPts, x, y)} className="line" stroke={color} />
        {hoverPt && (
          <g>
            <line
              x1={x(hoverPt.x)}
              x2={x(hoverPt.x)}
              y1={MARGIN.top}
              y2={MARGIN.top + innerH}
              className="crosshair"
            />
            <circle cx={x(hoverPt.x)} cy={y(hoverPt.y)} r={4} fill={color} />
          </g>
        )}
      </svg>
      <div className="chart-readout" data-testid="chart-readout">
        {hoverPt ? `${xFormat(hoverPt.x)} · ${yFormat(hoverPt.y)}` : " "}
      </div>
    </div>
  );
}
