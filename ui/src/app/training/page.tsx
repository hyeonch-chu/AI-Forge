'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';

// ── Constants ─────────────────────────────────────────────────────────────────

const YOLO_PRESETS = [
  // YOLOv8
  'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
  // YOLOv10
  'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt',
  // YOLO11
  'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
];

const EXPORT_FORMATS = ['none', 'onnx', 'engine', 'torchscript', 'coreml', 'saved_model'];

const POLL_INTERVAL_MS = 2000; // poll status every 2 s while job is running

// ── Types ─────────────────────────────────────────────────────────────────────

interface TrainJob {
  job_id: string;
  status: 'running' | 'done' | 'failed';
  exit_code: number | null;
  log_tail: string[];
}

// ── Sub-components ────────────────────────────────────────────────────────────

/** Colour-coded badge for job status. */
function StatusBadge({ status }: { status: string }) {
  const cls =
    status === 'running'
      ? 'bg-blue-900/50 text-blue-300 animate-pulse'
      : status === 'done'
        ? 'bg-green-900/50 text-green-400'
        : 'bg-red-900/50 text-red-400';
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status.toUpperCase()}
    </span>
  );
}

/** Labelled form field wrapper. */
function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-gray-400">{label}</label>
      {children}
      {hint && <p className="text-xs text-gray-600">{hint}</p>}
    </div>
  );
}

// Shared Tailwind classes for form inputs
const inputCls =
  'bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 ' +
  'placeholder-gray-600 focus:outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600';

/** Colour-code log lines by content for readability. */
function lineColor(line: string): string {
  if (/error|traceback|exception/i.test(line)) return 'text-red-400';
  if (/warning/i.test(line))                   return 'text-yellow-400';
  if (/^\d+\/\d+\s+\d/.test(line))             return 'text-cyan-300';   // epoch progress
  if (/mAP|precision|recall/i.test(line))      return 'text-green-400';  // val metrics
  if (/Downloading|Unzipping/i.test(line))     return 'text-blue-400';   // download
  return 'text-gray-300';
}

// ── Main page ─────────────────────────────────────────────────────────────────

/** Training page — submit a YOLO training job and watch live log output. */
export default function TrainingPage() {
  // ── Form state ──────────────────────────────────────────────────────────────
  const [data, setData] = useState('');
  const [model, setModel] = useState('yolov8n.pt');
  const [epochs, setEpochs] = useState(50);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(16);
  const [experiment, setExperiment] = useState('yolo_training');
  const [runName, setRunName] = useState('');
  const [device, setDevice] = useState('cpu');
  const [registerName, setRegisterName] = useState('');
  const [exportFormat, setExportFormat] = useState('none');

  // ── Job state ───────────────────────────────────────────────────────────────
  const [job, setJob] = useState<TrainJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Auto-scroll log to bottom
  const logRef = useRef<HTMLDivElement>(null);

  // ── Polling ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!job || job.status !== 'running') return;

    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/train/status/${job.job_id}`);
        if (!res.ok) return;
        const updated: TrainJob = await res.json();
        setJob(updated);
        // Auto-scroll log panel
        if (logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
      } catch {
        // Silently ignore transient network errors during polling
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(timer);
  }, [job]);

  // Scroll to bottom whenever log_tail grows
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [job?.log_tail.length]);

  // ── Submit handler ──────────────────────────────────────────────────────────
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    setSubmitting(true);
    setJob(null);

    const payload = {
      data,
      model,
      epochs,
      imgsz,
      batch,
      experiment: experiment || 'yolo_training',
      run_name: runName || null,
      device,
      register_name: registerName || null,
      export_format: exportFormat === 'none' ? null : exportFormat,
    };

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result: TrainJob | { error: string } = await res.json();
      if (!res.ok || 'error' in result) {
        setSubmitError(('error' in result ? result.error : null) ?? `HTTP ${res.status}`);
      } else {
        setJob(result);
      }
    } catch (err) {
      setSubmitError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div>
      {/* Page header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-100">Training</h2>
        <p className="text-gray-500 mt-1">
          Submit a YOLO training job — results are tracked automatically in MLflow
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Left: Training form ─────────────────────────────────────────────── */}
        <form
          onSubmit={handleSubmit}
          className="rounded-xl bg-gray-900 border border-gray-800 p-6 space-y-5"
        >
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            Job Parameters
          </h3>

          {/* Dataset YAML path — required */}
          <Field
            label="Dataset YAML *"
            hint="Absolute path to the dataset config inside the trainer container (e.g. /data/coco.yaml)"
          >
            <input
              className={inputCls}
              type="text"
              required
              value={data}
              onChange={(e) => setData(e.target.value)}
              placeholder="/data/dataset.yaml"
            />
          </Field>

          {/* Model preset */}
          <Field label="Model" hint="Ultralytics will auto-download the weights on first use">
            <select
              className={inputCls}
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              {YOLO_PRESETS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </Field>

          {/* Numeric params in a 3-column grid */}
          <div className="grid grid-cols-3 gap-3">
            <Field label="Epochs">
              <input
                className={inputCls}
                type="number"
                min={1}
                max={1000}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </Field>
            <Field label="Image Size">
              <input
                className={inputCls}
                type="number"
                min={32}
                max={1280}
                step={32}
                value={imgsz}
                onChange={(e) => setImgsz(Number(e.target.value))}
              />
            </Field>
            <Field label="Batch Size">
              <input
                className={inputCls}
                type="number"
                min={1}
                max={512}
                value={batch}
                onChange={(e) => setBatch(Number(e.target.value))}
              />
            </Field>
          </div>

          {/* Device + Export format in a 2-column grid */}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Device" hint="'cpu' or GPU index e.g. '0'">
              <input
                className={inputCls}
                type="text"
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                placeholder="cpu"
              />
            </Field>
            <Field label="Export Format" hint="Optional post-training export">
              <select
                className={inputCls}
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value)}
              >
                {EXPORT_FORMATS.map((f) => (
                  <option key={f} value={f}>
                    {f === 'none' ? '— none —' : f}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          {/* MLflow metadata */}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Experiment Name">
              <input
                className={inputCls}
                type="text"
                value={experiment}
                onChange={(e) => setExperiment(e.target.value)}
                placeholder="yolo_training"
              />
            </Field>
            <Field label="Run Name" hint="Optional">
              <input
                className={inputCls}
                type="text"
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                placeholder="baseline-run"
              />
            </Field>
          </div>

          {/* Register model */}
          <Field label="Register Model As" hint="Leave blank to skip model registry">
            <input
              className={inputCls}
              type="text"
              value={registerName}
              onChange={(e) => setRegisterName(e.target.value)}
              placeholder="yolo_detector"
            />
          </Field>

          {/* Submit error */}
          {submitError && (
            <p className="text-sm text-red-400 bg-red-900/20 border border-red-800 rounded-lg px-3 py-2">
              {submitError}
            </p>
          )}

          {/* Submit button */}
          <button
            type="submit"
            disabled={submitting || job?.status === 'running'}
            className={`w-full py-2.5 rounded-lg text-sm font-semibold transition-colors ${
              submitting || job?.status === 'running'
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {submitting ? 'Submitting…' : job?.status === 'running' ? 'Training in progress…' : 'Start Training'}
          </button>
        </form>

        {/* ── Right: Job status + log viewer ──────────────────────────────────── */}
        <div className="space-y-4">
          {/* No job yet */}
          {!job && !submitting && (
            <div className="rounded-xl bg-gray-900 border border-dashed border-gray-700 p-10 text-center text-gray-500 text-sm">
              Fill in the form and press <span className="text-gray-400 font-medium">Start Training</span> to launch a job.
            </div>
          )}

          {/* Job card */}
          {job && (
            <div className="rounded-xl bg-gray-900 border border-gray-800 p-5 space-y-4">
              {/* Header row */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-500 mb-1">Job ID</p>
                  <p className="font-mono text-sm text-gray-200">{job.job_id}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-500 mb-1">Status</p>
                  <StatusBadge status={job.status} />
                </div>
              </div>

              {/* Exit code on finish */}
              {job.exit_code !== null && (
                <p className="text-xs text-gray-500">
                  Exit code:{' '}
                  <span className={job.exit_code === 0 ? 'text-green-400' : 'text-red-400'}>
                    {job.exit_code}
                  </span>
                </p>
              )}

              {/* Done banner with link to Experiments */}
              {job.status === 'done' && (
                <div className="rounded-lg bg-green-900/20 border border-green-800 px-4 py-3 text-sm text-green-400 flex items-center justify-between">
                  <span>Training complete — results saved to MLflow.</span>
                  <Link
                    href="/experiments"
                    className="ml-3 text-xs underline hover:text-green-300"
                  >
                    View Experiments →
                  </Link>
                </div>
              )}

              {/* Failed banner */}
              {job.status === 'failed' && (
                <div className="rounded-lg bg-red-900/20 border border-red-800 px-4 py-3 text-sm text-red-400">
                  Training failed. Check the log below for details.
                </div>
              )}

              {/* Log viewer */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <p className="text-xs text-gray-500">
                    Output log {job.status === 'running' && `(live — ${job.log_tail.length} lines)`}
                  </p>
                  {job.status === 'running' && (
                    <span className="text-xs text-blue-400 animate-pulse">● live</span>
                  )}
                </div>
                <div
                  ref={logRef}
                  className="h-[560px] overflow-y-auto rounded-lg bg-gray-950 border border-gray-800 p-3 font-mono text-xs leading-5"
                >
                  {job.log_tail.length === 0 ? (
                    <span className="text-gray-600">Waiting for output…</span>
                  ) : (
                    job.log_tail.map((line, i) => (
                      <div key={i} className={`whitespace-pre-wrap break-all ${lineColor(line)}`}>
                        {line}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
