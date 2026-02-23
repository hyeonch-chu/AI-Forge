'use client';

import { useEffect, useState } from 'react';

/** A single model version from the MLflow registry. */
interface ModelVersion {
  name: string;
  version: string;
  current_stage: string;
  status: string;
  creation_timestamp: number;
}

/** A registered model from the MLflow model registry. */
interface RegisteredModel {
  name: string;
  latest_versions?: ModelVersion[];
  creation_timestamp: number;
  last_updated_timestamp: number;
  description?: string;
}

/** Tailwind classes for model stage badge styling. */
function stageClass(stage: string): string {
  switch (stage) {
    case 'Production':
      return 'bg-green-900/50 text-green-400';
    case 'Staging':
      return 'bg-yellow-900/50 text-yellow-400';
    case 'Archived':
      return 'bg-gray-800 text-gray-500';
    default:
      return 'bg-gray-800 text-gray-400';
  }
}

/** Model registry page — lists all registered MLflow models with version and stage info. */
export default function ModelsPage() {
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/models')
      .then((r) => r.json())
      .then((data) => {
        setModels(data.registered_models ?? []);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(String(err));
        setLoading(false);
      });
  }, []);

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-100">Models</h2>
        <p className="text-gray-500 mt-1">MLflow model registry — registered model versions and stages</p>
      </div>

      {loading && <p className="text-sm text-gray-500">Loading models…</p>}

      {error && (
        <div className="p-4 rounded-lg bg-red-900/30 border border-red-800 text-red-400 text-sm">
          Could not reach MLflow: {error}
        </div>
      )}

      {!loading && !error && models.length === 0 && (
        <div className="p-10 rounded-xl bg-gray-900 border border-dashed border-gray-700 text-center text-gray-500 text-sm">
          No registered models found. Train a model and register it to see it here.
        </div>
      )}

      <div className="space-y-4">
        {models.map((model) => (
          <div
            key={model.name}
            className="p-5 rounded-xl bg-gray-900 border border-gray-800 hover:border-gray-700 transition-colors"
          >
            {/* Model header */}
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-base font-semibold text-gray-100">{model.name}</h3>
                {model.description && (
                  <p className="text-sm text-gray-500 mt-1">{model.description}</p>
                )}
              </div>
              <span className="shrink-0 text-xs text-gray-600">
                Updated {new Date(model.last_updated_timestamp).toLocaleDateString()}
              </span>
            </div>

            {/* Latest versions */}
            {model.latest_versions && model.latest_versions.length > 0 && (
              <div className="mt-4">
                <p className="text-xs text-gray-600 mb-2">Latest versions</p>
                <div className="flex flex-wrap gap-2">
                  {model.latest_versions.map((v) => (
                    <div
                      key={`${v.name}-${v.version}`}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-800 text-sm"
                    >
                      <span className="text-gray-300 font-mono text-xs">v{v.version}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${stageClass(v.current_stage)}`}>
                        {v.current_stage}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
