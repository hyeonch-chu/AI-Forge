// ui/src/app/api/train/status/[jobId]/route.ts
// Next.js API route — proxies GET /api/train/status/:jobId
//                              → trainer:6006/api/v1/train/status/:jobId

import { NextRequest, NextResponse } from 'next/server';

const TRAIN_URL = process.env.TRAIN_URL ?? 'http://trainer:6006';

export async function GET(
  _req: NextRequest,
  { params }: { params: { jobId: string } },
): Promise<NextResponse> {
  try {
    const upstream = await fetch(
      `${TRAIN_URL}/api/v1/train/status/${params.jobId}`,
    );
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    console.error('[api/train/status] trainer unreachable:', err);
    return NextResponse.json({ error: 'Train service unreachable' }, { status: 503 });
  }
}
