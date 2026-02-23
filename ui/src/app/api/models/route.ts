/** Proxy: GET /api/models → MLflow /api/2.0/mlflow/registered-models/search */

const MLFLOW_URL = process.env.MLFLOW_TRACKING_URI ?? 'http://mlflow:5000';

export async function GET() {
  try {
    const res = await fetch(
      `${MLFLOW_URL}/api/2.0/mlflow/registered-models/search?max_results=100`,
      { next: { revalidate: 30 } }
    );
    if (!res.ok) {
      return Response.json({ registered_models: [] }, { status: res.status });
    }
    const data = await res.json();
    return Response.json(data);
  } catch (err) {
    return Response.json(
      { error: `Cannot reach MLflow: ${err}`, registered_models: [] },
      { status: 502 }
    );
  }
}
