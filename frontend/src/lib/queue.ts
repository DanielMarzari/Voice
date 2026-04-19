// Client for Voice Studio's render-queue endpoints. The Queue tab polls
// these to render the worker status card.

import { u } from "./api";

export type QueueSnapshot = {
  running: boolean;
  paused: boolean;
  overnight_only: boolean;
  reader_configured: boolean;
  last_poll_at: string | null;
  last_error: string | null;
  current: null | {
    job_id: string;
    voice_name: string | null;
    document_title: string | null;
    chunks_total: number | null;
    chunks_done: number;
  };
  jobs_completed: number;
  jobs_failed: number;
  settings: {
    render_worker_enabled: boolean;
    overnight_only: boolean;
  };
};

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText}${body ? `: ${body.slice(0, 300)}` : ""}`);
  }
  return r.json();
}

export async function getQueue(): Promise<QueueSnapshot> {
  return j(await fetch(u("/api/render-queue")));
}

export async function pauseQueue(): Promise<QueueSnapshot> {
  return j(await fetch(u("/api/render-queue/pause"), { method: "POST" }));
}

export async function resumeQueue(): Promise<QueueSnapshot> {
  return j(await fetch(u("/api/render-queue/resume"), { method: "POST" }));
}

export async function updateQueueSettings(patch: {
  render_worker_enabled?: boolean;
  overnight_only?: boolean;
}): Promise<QueueSnapshot> {
  return j(
    await fetch(u("/api/render-queue/settings"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  );
}

export async function testQueueConnection(): Promise<{
  ok: boolean;
  pending_count: number;
}> {
  return j(await fetch(u("/api/render-queue/test"), { method: "POST" }));
}
