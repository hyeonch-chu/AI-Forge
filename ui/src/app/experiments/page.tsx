'use client';

import { useEffect, useState } from 'react';
import MetricsChart, { type ChartRun } from '@/components/MetricsChart';

/** MLflow experiment record returned by the search API. */
interface Experiment {
  experiment_id: string;
  name: string;
  lifecycle_stage: string;
  creation_time: number;
  last_update_time: number;
}

/** MLflow run record (simplified) returned by the runs/search API. */
interface Run {
  info: {
    run_id: string;
    run_name: string;
    status: string;
    start_time: number;
    end_time?: number;
  };
  data: {
    metrics: Record<string, number>;
    params: Record<string, string>;
  };
}

/** Tailwind badge class based on MLflow run status. */
function statusClass(status: string): string {
  switch (status) {
    case 'FINISHED':
      return 'bg-green-900/50 text-green-400';
    case 'RUNNING':
      return 'bg-blue-900/50 text-blue-400';
    case 'FAILED':
      return 'bg-red-900/50 text-red-400';
    default:
      return 'bg-gray-800 text-gray-400';
  }
}

/** Format a Unix-millisecond timestamp to a locale string. */
function fmt(ms: number): string {
  return new Date(ms).toLocaleString();
}

/** Convert an array of MLflow Run records to MetricsChart data. */
function toChartRuns(runs: Run[]): ChartRun[] {
  return runs.map((r) => ({
    id: r.info.run_id,
    name: r.info.run_name || '',
    map50: r.data.metrics['metrics/mAP50'],
    map5095: r.data.metrics['metrics/mAP50-95'],
  }));
}

/** Experiment tracking dashboard — lists MLflow experiments and their runs. */
export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [runsLoading, setRunsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/experiments')
      .then((r) => r.json())
      .then((data) => {
        // Exclude the built-in "Default" experiment and soft-deleted ones
        const active = (data.experiments ?? []).filter(
          (e: Experiment) => e.name !== 'Default' && e.lifecycle_stage === 'active'
        );
        setExperiments(active);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(String(err));
        setLoading(false);
      });
  }, []);

  /** Toggle the runs panel for an experiment — fetches runs on first open. */
  async function toggleRuns(expId: string) {
    if (expanded === expId) {
      setExpanded(null);
      setRuns([]);
      return;
    }
    setExpanded(expId);
    setRunsLoading(true);
    try {
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: [expId], max_results: 25 }),
      });
      const data = await res.json();
      setRuns(data.runs ?? []);
    } catch (err) {
      console.error('Failed to fetch runs:', err);
    } finally {
      setRunsLoading(false);
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-100">Experiments</h2>
        <p className="text-gray-500 mt-1">MLflow experiment tracking — click a row to view runs</p>
      </div>

      {loading && <p className="text-sm text-gray-500">Loading experiments…</p>}

      {error && (
        <div className="p-4 rounded-lg bg-red-900/30 border border-red-800 text-red-400 text-sm">
          Could not reach MLflow: {error}
        </div>
      )}

      {!loading && !error && experiments.length === 0 && (
        <div className="p-10 rounded-xl bg-gray-900 border border-dashed border-gray-700 text-center text-gray-500 text-sm">
          No experiments found. Run a training job to create one.
        </div>
      )}

      <div className="space-y-3">
        {experiments.map((exp) => (
          <div
            key={exp.experiment_id}
            className="rounded-xl bg-gray-900 border border-gray-800 overflow-hidden"
          >
            {/* Experiment header — click to expand */}
            <button
              onClick={() => toggleRuns(exp.experiment_id)}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-800/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="text-base font-medium text-gray-100">{exp.name}</span>
                <span className="text-xs text-gray-600">#{exp.experiment_id}</span>
              </div>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span>Updated {fmt(exp.last_update_time)}</span>
                <span>{expanded === exp.experiment_id ? '▲' : '▼'}</span>
              </div>
            </button>

            {/* Expanded panel: metrics chart + runs table */}
            {expanded === exp.experiment_id && (
              <div className="border-t border-gray-800 p-4">
                {runsLoading && <p className="text-sm text-gray-500">Loading runs…</p>}

                {!runsLoading && runs.length === 0 && (
                  <p className="text-sm text-gray-500">No runs found for this experiment.</p>
                )}

                {!runsLoading && runs.length > 0 && (
                  <>
                    {/* Metrics bar chart (only rendered when there are metric values) */}
                    <MetricsChart runs={toChartRuns(runs)} />

                    {/* Runs table */}
                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                            <th className="pb-2 pr-4">Run Name</th>
                            <th className="pb-2 pr-4">Status</th>
                            <th className="pb-2 pr-4">Started</th>
                            <th className="pb-2 pr-4">mAP50</th>
                            <th className="pb-2">mAP50-95</th>
                          </tr>
                        </thead>
                        <tbody>
                          {runs.map((run) => (
                            <tr
                              key={run.info.run_id}
                              className="border-b border-gray-800/40 hover:bg-gray-800/30 transition-colors"
                            >
                              <td className="py-2 pr-4 text-gray-300">
                                {run.info.run_name || <span className="text-gray-600">—</span>}
                              </td>
                              <td className="py-2 pr-4">
                                <span
                                  className={`text-xs px-2 py-0.5 rounded ${statusClass(run.info.status)}`}
                                >
                                  {run.info.status}
                                </span>
                              </td>
                              <td className="py-2 pr-4 text-gray-500 text-xs">
                                {fmt(run.info.start_time)}
                              </td>
                              <td className="py-2 pr-4 text-gray-300 font-mono text-xs">
                                {run.data.metrics['metrics/mAP50'] !== undefined
                                  ? run.data.metrics['metrics/mAP50'].toFixed(4)
                                  : '—'}
                              </td>
                              <td className="py-2 text-gray-300 font-mono text-xs">
                                {run.data.metrics['metrics/mAP50-95'] !== undefined
                                  ? run.data.metrics['metrics/mAP50-95'].toFixed(4)
                                  : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
