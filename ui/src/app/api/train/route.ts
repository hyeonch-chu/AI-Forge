// ui/src/app/api/train/route.ts
// Next.js API route — proxies POST /api/train → trainer:6006/api/v1/train
// The trainer service is reachable only on the Docker 'backend' network;
// this server-side route acts as the bridge for the browser.

import { NextRequest, NextResponse } from 'next/server';

const TRAIN_URL = process.env.TRAIN_URL ?? 'http://trainer:6006';

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${TRAIN_URL}/api/v1/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data: unknown = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    console.error('[api/train] trainer unreachable:', err);
    return NextResponse.json({ error: 'Train service unreachable' }, { status: 503 });
  }
}
