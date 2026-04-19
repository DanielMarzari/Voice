"use client";

// QueueTab — status + control for the render worker that drains Reader's
// render_jobs table. Shows what the worker is currently rendering, lets
// the user pause/resume, toggle overnight-only mode, and run a one-click
// connectivity test to confirm the bearer token works.

import { useCallback, useEffect, useState } from "react";
import {
  getQueue,
  pauseQueue,
  resumeQueue,
  testQueueConnection,
  updateQueueSettings,
  type QueueSnapshot,
} from "@/lib/queue";

export function QueueTab() {
  const [snapshot, setSnapshot] = useState<QueueSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await getQueue();
      setSnapshot(s);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  // Initial + polling.
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  async function handlePause() {
    setSnapshot(await pauseQueue());
  }
  async function handleResume() {
    setSnapshot(await resumeQueue());
  }
  async function handleOvernightToggle(next: boolean) {
    setSnapshot(await updateQueueSettings({ overnight_only: next }));
  }
  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await testQueueConnection();
      setTestResult(`✓ ${r.pending_count} pending job(s) visible`);
    } catch (e) {
      setTestResult(`✗ ${(e as Error).message}`);
    } finally {
      setTesting(false);
    }
  }

  if (!snapshot) {
    return (
      <div className="text-sm text-[color:var(--muted)]">
        {error ?? "Loading worker status…"}
      </div>
    );
  }

  const { current, settings } = snapshot;
  const progressPct =
    current && current.chunks_total
      ? Math.round((current.chunks_done / current.chunks_total) * 100)
      : null;

  return (
    <div className="space-y-6 max-w-[720px]">
      {/* Status card */}
      <div className="card">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-semibold">Render worker</div>
          <span
            className="chip"
            title={snapshot.running ? "Running" : "Stopped"}
          >
            {snapshot.paused ? "● paused" : snapshot.running ? "● running" : "● stopped"}
          </span>
        </div>
        <div className="text-xs text-[color:var(--muted)] mb-3">
          Polls Reader for pending render jobs. When the Queue has work, the
          worker claims one, synthesizes locally with F5-TTS or XTTS-v2,
          and ships MP3 chunks back. Reader never waits on your Mac to
          initiate anything — this is pull-only.
        </div>

        {current ? (
          <div className="bg-[color:var(--surface-2)] rounded-lg p-3 mb-3">
            <div className="text-sm font-medium">
              Rendering: {current.document_title ?? "(untitled)"}
            </div>
            <div className="text-xs text-[color:var(--muted)] mt-0.5">
              Voice: {current.voice_name ?? "?"}
              {current.chunks_total != null && (
                <> · Chunk {current.chunks_done}/{current.chunks_total}</>
              )}
            </div>
            {progressPct != null && (
              <div className="h-1.5 rounded-full overflow-hidden mt-2" style={{ background: "var(--surface)" }}>
                <div
                  className="h-full rounded-full"
                  style={{ width: `${progressPct}%`, background: "var(--accent)" }}
                />
              </div>
            )}
          </div>
        ) : (
          <div className="text-sm text-[color:var(--muted)] mb-3">
            {snapshot.paused
              ? "Worker paused — click Resume to start draining the queue."
              : settings.overnight_only && !isOvernight()
              ? "Idle (overnight-only mode; waits until 10:00 PM local)."
              : !snapshot.reader_configured
              ? "Reader not configured — paste your token in the Settings gear."
              : "Idle — waiting for jobs."}
          </div>
        )}

        <div className="flex items-center gap-3 text-xs text-[color:var(--muted)]">
          <span>Last poll: {snapshot.last_poll_at ?? "—"}</span>
          <span>Completed: {snapshot.jobs_completed}</span>
          <span>Failed: {snapshot.jobs_failed}</span>
        </div>

        {snapshot.last_error && (
          <div className="mt-3 text-xs text-red-500 bg-red-500/10 rounded px-2 py-1.5 break-all">
            {snapshot.last_error}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="card">
        <div className="text-sm font-semibold mb-3">Controls</div>

        <div className="flex items-center gap-2 mb-4 flex-wrap">
          {snapshot.paused ? (
            <button className="btn btn-primary" onClick={handleResume}>
              ▶ Resume
            </button>
          ) : (
            <button className="btn" onClick={handlePause}>
              ⏸ Pause
            </button>
          )}
          <button
            className="btn"
            onClick={handleTest}
            disabled={testing}
            title="Hit Reader with the stored bearer token to confirm auth"
          >
            {testing ? "Testing…" : "Test Reader connection"}
          </button>
          {testResult && (
            <span className="text-xs text-[color:var(--muted)]">{testResult}</span>
          )}
        </div>

        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            className="mt-1"
            checked={settings.overnight_only}
            onChange={(e) => handleOvernightToggle(e.target.checked)}
          />
          <span>
            <div className="text-sm font-medium">Overnight only</div>
            <div className="text-xs text-[color:var(--muted)]">
              Worker stays idle 6 AM – 10 PM local. Pending jobs pile up and
              burn through overnight. Good for long documents when you don&apos;t
              want the fans spinning during the workday.
            </div>
          </span>
        </label>
      </div>

      <div className="text-xs text-[color:var(--muted)]">
        Tip: queue renders from Reader — open a doc and hit{" "}
        <strong>▶ Listen with…</strong> in the top-right.
      </div>
    </div>
  );
}

function isOvernight(): boolean {
  const h = new Date().getHours();
  return h >= 22 || h < 6;
}
