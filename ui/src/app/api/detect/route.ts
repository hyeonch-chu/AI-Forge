/** Proxy: POST /api/detect → Inference service POST /api/v1/detect */

const INFERENCE_URL = process.env.INFERENCE_URL ?? 'http://inference:8000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const res = await fetch(`${INFERENCE_URL}/api/v1/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch (err) {
    return Response.json(
      { success: false, error: `Cannot reach inference service: ${err}` },
      { status: 502 }
    );
  }
}
