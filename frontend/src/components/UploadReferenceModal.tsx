"use client";

/**
 * UploadReferenceModal — the "Upload" entry point on the Design tab's
 * Base voice row. Replaces the prior prompt() shim with a proper dialog
 * that also captures a short description so the dropdown shows
 * "Alex (friendly, inviting and balanced)".
 */

import { useEffect, useRef, useState } from "react";
import { DropZone } from "@/components/DropZone";

type Props = {
  open: boolean;
  onClose: () => void;
  onSubmit: (args: { name: string; description: string; file: File }) => Promise<void>;
};

export function UploadReferenceModal({ open, onClose, onSubmit }: Props) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setName("");
    setDescription("");
    setFile(null);
    setError(null);
    // Auto-focus the name field when the modal opens.
    setTimeout(() => nameRef.current?.focus(), 30);
  }, [open]);

  if (!open) return null;

  async function handleSubmit() {
    if (!name.trim() || !file || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim(),
        file,
      });
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
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
          <h2 className="text-base font-semibold">Upload a reference voice</h2>
          <button
            onClick={onClose}
            className="text-[color:var(--muted)] hover:text-[color:var(--foreground)] text-lg"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <p className="text-xs text-[color:var(--muted)] mb-4">
          Drop in a clean ~10-second speech clip of any voice. Any common
          format works (WAV/MP3/M4A/OGG/FLAC) — it gets transcoded to 24
          kHz mono WAV for synthesis.
        </p>

        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium mb-1">
              Name <span className="text-[color:var(--muted)]">(the person)</span>
            </label>
            <input
              ref={nameRef}
              className="input w-full"
              placeholder="Alex"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={60}
              onKeyDown={(e) => {
                if (e.key === "Enter" && name.trim() && file) handleSubmit();
              }}
            />
          </div>

          <div>
            <label className="block text-xs font-medium mb-1">
              Description{" "}
              <span className="text-[color:var(--muted)]">
                (shows next to the name in the dropdown)
              </span>
            </label>
            <input
              className="input w-full"
              placeholder="Friendly, inviting and balanced"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={120}
            />
          </div>

          <div>
            <label className="block text-xs font-medium mb-1">Audio file</label>
            <DropZone
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
              file={file}
              onFile={setFile}
              label="Drag & drop a voice clip"
              hint="~10 s of clean speech works best"
              minHeight={110}
            />
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 mt-6">
          {error && (
            <span className="text-xs text-red-500 flex-1 break-all">{error}</span>
          )}
          <button className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={busy || !name.trim() || !file}
          >
            {busy ? "Uploading…" : "Upload"}
          </button>
        </div>
      </div>
    </div>
  );
}
