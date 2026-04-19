"use client";

/**
 * DropZone — a drag-and-drop area that also offers click-to-pick.
 *
 * Used by ImportTab (audio + cover) and UploadReferenceModal (audio).
 * Intentionally does NOT auto-open the file picker when the card is
 * rendered — the user complained about that. The picker opens ONLY
 * when the small "Browse" button inside the zone is clicked; dragging
 * a file over and dropping it is the primary path.
 */

import { useCallback, useRef, useState } from "react";

type Props = {
  accept: string;           // HTML `accept` attribute, e.g. "audio/*,.wav,.mp3"
  file: File | null;
  onFile: (file: File | null) => void;
  label: string;            // main prompt ("Drop an audio file here")
  hint?: string;            // secondary line ("WAV/MP3/M4A · up to 20 MB")
  minHeight?: number;       // in pixels
  /** Optional inline preview element (e.g. <audio controls src={blobUrl} />)
   *  rendered under the file info when a file is selected. */
  preview?: React.ReactNode;
};

export function DropZone({
  accept,
  file,
  onFile,
  label,
  hint,
  minHeight = 120,
  preview,
}: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    // Must preventDefault on dragover for drop to fire.
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }, []);

  return (
    <div
      className={[
        "card transition-colors select-none",
        dragging ? "!border-[color:var(--accent)] bg-[color:var(--surface-2)]" : "",
      ].join(" ")}
      style={{ minHeight }}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragEnter={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {!file ? (
        <div className="flex flex-col items-center justify-center text-center py-4 h-full">
          <div className="text-3xl opacity-50 mb-1" aria-hidden>
            ⇣
          </div>
          <div className="text-sm font-medium">{label}</div>
          {hint && (
            <div className="text-xs text-[color:var(--muted)] mt-1">{hint}</div>
          )}
          <button
            type="button"
            className="btn mt-3 !py-1.5 !px-3 text-xs"
            onClick={() => inputRef.current?.click()}
          >
            Browse…
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-stretch py-1">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium truncate" title={file.name}>
                {file.name}
              </div>
              <div className="text-xs text-[color:var(--muted)] mt-0.5">
                {formatSize(file.size)}
              </div>
            </div>
            <div className="flex gap-1 flex-shrink-0">
              <button
                type="button"
                className="btn !py-1 !px-2 text-xs"
                onClick={() => inputRef.current?.click()}
                title="Replace with a different file"
              >
                Replace
              </button>
              <button
                type="button"
                className="btn !py-1 !px-2 text-xs !text-red-500"
                onClick={() => onFile(null)}
                aria-label="Clear file"
                title="Clear"
              >
                ✕
              </button>
            </div>
          </div>
          {preview && <div className="mt-3">{preview}</div>}
        </div>
      )}
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
