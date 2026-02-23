/** Proxy: POST /api/runs → MLflow /api/2.0/mlflow/runs/search */

const MLFLOW_URL = process.env.MLFLOW_TRACKING_URI ?? 'http://mlflow:5000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const res = await fetch(`${MLFLOW_URL}/api/2.0/mlflow/runs/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch (err) {
    return Response.json(
      { error: `Cannot reach MLflow: ${err}`, runs: [] },
      { status: 502 }
    );
  }
}
