'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

/** A single detection prediction from the inference API. */
interface Prediction {
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] pixel coords
}

/** Response schema from POST /api/v1/detect. */
interface DetectResponse {
  success: boolean;
  predictions: Prediction[];
  metrics: {
    latency_ms: number;
    image_width: number;
    image_height: number;
    num_predictions: number;
  };
  error?: string;
}

/** Distinct colors for rendering up to 10 bounding-box classes. */
const BOX_COLORS = [
  '#3b82f6',
  '#ef4444',
  '#22c55e',
  '#f59e0b',
  '#8b5cf6',
  '#ec4899',
  '#14b8a6',
  '#f97316',
  '#6366f1',
  '#a3e635',
];

/** Draw bounding boxes and labels on a canvas element. */
function renderBoxes(
  canvas: HTMLCanvasElement,
  imageUrl: string,
  predictions: Prediction[]
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const img = new Image();
  img.onload = () => {
    // Match canvas pixel dimensions to the image so boxes align correctly
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    predictions.forEach((pred, i) => {
      const [x1, y1, x2, y2] = pred.bbox;
      const color = BOX_COLORS[i % BOX_COLORS.length];
      const labelText = `${pred.label} ${(pred.confidence * 100).toFixed(1)}%`;

      // Box stroke
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, img.naturalWidth / 320);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label background
      ctx.font = `${Math.max(12, img.naturalWidth / 53)}px sans-serif`;
      const textW = ctx.measureText(labelText).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - (Math.max(12, img.naturalWidth / 53) + 6), textW + 8, Math.max(12, img.naturalWidth / 53) + 6);

      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x1 + 4, y1 - 4);
    });
  };
  img.src = imageUrl;
}

/** Inference UI — upload an image, run detection, and visualize results. */
export default function InferencePage() {
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Re-draw boxes whenever results or source image changes
  const drawBoxes = useCallback(() => {
    if (!canvasRef.current || !imageUrl || !result) return;
    renderBoxes(canvasRef.current, imageUrl, result.predictions);
  }, [imageUrl, result]);

  useEffect(() => {
    if (result && imageUrl) drawBoxes();
  }, [result, imageUrl, drawBoxes]);

  /** Handle file selection: create object URL for preview and read base64. */
  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setResult(null);
    setError(null);

    // Object URL for canvas rendering (avoids double base64 decode)
    const url = URL.createObjectURL(file);
    setImageUrl(url);

    // Read as data URL, then strip the prefix to get raw base64
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setImageBase64(dataUrl.split(',')[1]);
    };
    reader.readAsDataURL(file);
  }

  /** Send the selected image to the detection API. */
  async function handleDetect() {
    if (!imageBase64) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 }),
      });
      const data: DetectResponse = await res.json();

      if (!res.ok || !data.success) {
        setError(data.error ?? `Detection failed (HTTP ${res.status})`);
      } else {
        setResult(data);
      }
    } catch (err: unknown) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-100">Inference</h2>
        <p className="text-gray-500 mt-1">Upload an image and run YOLO object detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Left column: upload + metrics ── */}
        <div className="space-y-4">
          {/* Upload card */}
          <div className="p-5 rounded-xl bg-gray-900 border border-gray-800">
            <p className="text-sm font-medium text-gray-400 mb-3">Image Upload</p>

            {/* Drop-zone / file picker */}
            <label className="flex flex-col items-center justify-center h-40 rounded-lg border-2 border-dashed border-gray-700 hover:border-blue-600 cursor-pointer transition-colors">
              <span className="text-3xl mb-2">📷</span>
              <span className="text-sm text-gray-500">Click to select an image</span>
              <span className="text-xs text-gray-700 mt-1">JPEG or PNG</span>
              <input
                type="file"
                accept="image/jpeg,image/png"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            {/* Thumbnail preview before detection */}
            {imageUrl && !result && (
              <div className="mt-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageUrl}
                  alt="Selected image preview"
                  className="w-full max-h-60 object-contain rounded-lg bg-gray-950"
                />
              </div>
            )}

            {/* Detect button */}
            <button
              onClick={handleDetect}
              disabled={!imageBase64 || loading}
              className="mt-4 w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium transition-colors"
            >
              {loading ? 'Running detection…' : 'Run Detection'}
            </button>
          </div>

          {/* Error banner */}
          {error && (
            <div className="p-4 rounded-lg bg-red-900/30 border border-red-800 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* No-model hint (empty predictions + no error = model not loaded) */}
          {result && result.predictions.length === 0 && !error && (
            <div className="p-4 rounded-lg bg-yellow-900/20 border border-yellow-800/50 text-yellow-600 text-sm">
              No objects detected. If no model is registered in MLflow the inference service
              returns empty predictions.
            </div>
          )}

          {/* Metrics card */}
          {result && (
            <div className="p-5 rounded-xl bg-gray-900 border border-gray-800">
              <p className="text-sm font-medium text-gray-400 mb-3">Metrics</p>
              <dl className="grid grid-cols-2 gap-3 text-sm">
                {(
                  [
                    ['Latency', `${result.metrics.latency_ms} ms`],
                    ['Detections', result.metrics.num_predictions],
                    ['Width', `${result.metrics.image_width} px`],
                    ['Height', `${result.metrics.image_height} px`],
                  ] as [string, string | number][]
                ).map(([k, v]) => (
                  <div key={k} className="bg-gray-800 rounded-lg px-3 py-2.5">
                    <dt className="text-xs text-gray-500">{k}</dt>
                    <dd className="text-gray-100 font-mono mt-0.5">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>

        {/* ── Right column: canvas + predictions table ── */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Detection canvas */}
              <div className="p-5 rounded-xl bg-gray-900 border border-gray-800">
                <p className="text-sm font-medium text-gray-400 mb-3">Detection Result</p>
                <canvas
                  ref={canvasRef}
                  className="w-full rounded-lg bg-gray-950"
                  style={{ maxHeight: '420px', objectFit: 'contain' }}
                />
              </div>

              {/* Predictions table */}
              <div className="p-5 rounded-xl bg-gray-900 border border-gray-800">
                <p className="text-sm font-medium text-gray-400 mb-3">
                  Predictions ({result.predictions.length})
                </p>

                {result.predictions.length === 0 ? (
                  <p className="text-sm text-gray-600">No objects detected.</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                          <th className="pb-2 pr-3">#</th>
                          <th className="pb-2 pr-3">Label</th>
                          <th className="pb-2 pr-3">Confidence</th>
                          <th className="pb-2">Bounding Box</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.predictions.map((pred, i) => (
                          <tr
                            key={i}
                            className="border-b border-gray-800/40 hover:bg-gray-800/30 transition-colors"
                          >
                            <td className="py-2 pr-3">
                              <span
                                className="inline-block w-2.5 h-2.5 rounded-full"
                                style={{ backgroundColor: BOX_COLORS[i % BOX_COLORS.length] }}
                              />
                            </td>
                            <td className="py-2 pr-3 text-gray-300">{pred.label}</td>
                            <td className="py-2 pr-3 text-gray-300">
                              {(pred.confidence * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 text-gray-500 font-mono text-xs">
                              [{pred.bbox.map((v) => Math.round(v)).join(', ')}]
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="h-full min-h-64 flex items-center justify-center rounded-xl bg-gray-900 border border-dashed border-gray-800 text-gray-600 text-sm">
              Detection results will appear here
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
