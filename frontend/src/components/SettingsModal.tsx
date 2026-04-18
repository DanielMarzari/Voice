"use client";

/**
 * SettingsModal — triggered by the gear icon in the header.
 * Lets the user paste their Reader base URL + auth token without
 * editing .env.local or restarting the backend. POSTs to /api/config.
 */

import { useEffect, useState } from "react";
import { u } from "@/lib/api";

type Props = {
  open: boolean;
  onClose: () => void;
  /** Called after the user saves so the header HealthPill can re-fetch. */
  onSaved: () => void;
  initialBaseUrl: string | null;
  tokenAlreadySet: boolean;
};

export function SettingsModal({
  open,
  onClose,
  onSaved,
  initialBaseUrl,
  tokenAlreadySet,
}: Props) {
  const [baseUrl, setBaseUrl] = useState("");
  const [token, setToken] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ok, setOk] = useState(false);

  useEffect(() => {
    if (open) {
      setBaseUrl(initialBaseUrl ?? "");
      setToken("");
      setError(null);
      setOk(false);
    }
  }, [open, initialBaseUrl]);

  if (!open) return null;

  async function save() {
    setSaving(true);
    setError(null);
    setOk(false);
    try {
      const r = await fetch(u("/api/config"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reader_base_url: baseUrl.trim() || null,
          // Empty token field = keep existing (don't clobber).
          reader_auth_token: token.trim() || null,
        }),
      });
      if (!r.ok) {
        throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
      }
      setOk(true);
      onSaved();
      setTimeout(() => onClose(), 600);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="card w-full max-w-md m-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold">Connect to Reader</h2>
          <button
            onClick={onClose}
            className="text-[color:var(--muted)] hover:text-[color:var(--foreground)] text-lg"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <p className="text-xs text-[color:var(--muted)] mb-4">
          These override anything in <code>.env.local</code> for the life of
          this backend process. Restart the backend to revert.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium mb-1">
              Reader base URL
            </label>
            <input
              className="input"
              placeholder="https://reader.danmarzari.com"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-xs font-medium mb-1">
              Studio token
              {tokenAlreadySet && (
                <span className="ml-2 text-[color:var(--muted)] font-normal">
                  (set — leave blank to keep)
                </span>
              )}
            </label>
            <input
              className="input"
              type="password"
              placeholder={
                tokenAlreadySet ? "••••••••••••••••" : "Paste from Reader → Voice Lab"
              }
              value={token}
              onChange={(e) => setToken(e.target.value)}
              autoComplete="off"
            />
            <div className="text-[11px] text-[color:var(--muted)] mt-1">
              Generate a token from Reader → Voice Lab → &quot;Connect Voice Studio&quot;.
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 mt-6">
          {error && <span className="text-xs text-red-500 flex-1">{error}</span>}
          {ok && !error && (
            <span className="text-xs text-green-600 flex-1">Saved.</span>
          )}
          <button className="btn" onClick={onClose} disabled={saving}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
