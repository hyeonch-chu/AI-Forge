/** Proxy: GET /api/experiments → MLflow /api/2.0/mlflow/experiments/search */

const MLFLOW_URL = process.env.MLFLOW_TRACKING_URI ?? 'http://mlflow:5000';

export async function GET() {
  try {
    const res = await fetch(
      `${MLFLOW_URL}/api/2.0/mlflow/experiments/search?max_results=200`,
      // Revalidate every 30 s to keep experiment list reasonably fresh
      { next: { revalidate: 30 } }
    );
    if (!res.ok) {
      return Response.json({ experiments: [] }, { status: res.status });
    }
    const data = await res.json();
    return Response.json(data);
  } catch (err) {
    return Response.json(
      { error: `Cannot reach MLflow: ${err}`, experiments: [] },
      { status: 502 }
    );
  }
}
