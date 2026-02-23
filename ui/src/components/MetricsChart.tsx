'use client';

/** Data point supplied to the chart for a single MLflow run. */
export interface ChartRun {
  id: string;
  name: string;
  map50?: number;
  map5095?: number;
}

interface MetricsChartProps {
  runs: ChartRun[];
}

// ── Chart layout constants ────────────────────────────────────────────────────
const W = 520; // viewBox width
const H = 200; // viewBox height
const PAD = { top: 12, right: 16, bottom: 50, left: 42 } as const;
const CW = W - PAD.left - PAD.right; // usable chart width
const CH = H - PAD.top - PAD.bottom; // usable chart height

// Y-axis reference values (metric range is always 0–1)
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1.0];

/** Convert a [0, 1] metric value to an SVG y-coordinate within the chart area. */
function yPos(v: number): number {
  return PAD.top + (1 - v) * CH;
}

/**
 * Pure SVG dual-bar chart — no external charting library.
 * Shows mAP50 (blue) and mAP50-95 (purple) side-by-side for the last N runs.
 * Renders nothing when no runs have metric data.
 */
export default function MetricsChart({ runs }: MetricsChartProps) {
  // Only use runs that have mAP50 data; cap at 10 for readability
  const data = runs.filter((r) => r.map50 !== undefined).slice(0, 10);

  if (data.length === 0) return null;

  const groupW = CW / data.length;
  const barW = Math.min(groupW * 0.32, 18);
  const barGap = barW * 0.4;

  return (
    <div className="mt-4 rounded-lg bg-gray-950 border border-gray-800 p-3">
      <p className="text-xs text-gray-500 mb-2">Detection metrics across runs</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Metrics bar chart">
        {/* ── Y-axis grid lines and labels ── */}
        {Y_TICKS.map((v) => (
          <g key={v}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={yPos(v)}
              y2={yPos(v)}
              stroke={v === 0 ? '#4b5563' : '#1f2937'}
              strokeWidth={v === 0 ? 1 : 0.5}
            />
            <text
              x={PAD.left - 5}
              y={yPos(v) + 3.5}
              textAnchor="end"
              fontSize={9}
              fill="#6b7280"
            >
              {v.toFixed(2)}
            </text>
          </g>
        ))}

        {/* ── Bars per run ── */}
        {data.map((run, i) => {
          const cx = PAD.left + i * groupW + groupW / 2;
          const map50 = run.map50 ?? 0;
          const map5095 = run.map5095 ?? 0;
          // Truncate long run names so x-axis labels don't overlap
          const label = (run.name || `run${i + 1}`).slice(0, 9);

          return (
            <g key={run.id}>
              {/* mAP50 bar */}
              {map50 > 0 && (
                <rect
                  x={cx - barW - barGap / 2}
                  y={yPos(map50)}
                  width={barW}
                  height={map50 * CH}
                  fill="#3b82f6"
                  opacity={0.9}
                  rx={2}
                >
                  <title>{`mAP50: ${map50.toFixed(4)}`}</title>
                </rect>
              )}

              {/* mAP50-95 bar */}
              {map5095 > 0 && (
                <rect
                  x={cx + barGap / 2}
                  y={yPos(map5095)}
                  width={barW}
                  height={map5095 * CH}
                  fill="#8b5cf6"
                  opacity={0.9}
                  rx={2}
                >
                  <title>{`mAP50-95: ${map5095.toFixed(4)}`}</title>
                </rect>
              )}

              {/* X-axis run label */}
              <text
                x={cx}
                y={PAD.top + CH + 15}
                textAnchor="middle"
                fontSize={9}
                fill="#6b7280"
              >
                {label}
              </text>
            </g>
          );
        })}

        {/* ── Legend ── */}
        <g transform={`translate(${PAD.left}, ${PAD.top + CH + 32})`}>
          <rect width={8} height={8} y={-1} fill="#3b82f6" opacity={0.9} rx={1} />
          <text x={11} y={7} fontSize={9} fill="#9ca3af">
            mAP50
          </text>
          <rect x={55} width={8} height={8} y={-1} fill="#8b5cf6" opacity={0.9} rx={1} />
          <text x={66} y={7} fontSize={9} fill="#9ca3af">
            mAP50-95
          </text>
        </g>
      </svg>
    </div>
  );
}
