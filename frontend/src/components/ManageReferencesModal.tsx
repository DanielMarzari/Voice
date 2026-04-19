"use client";

/**
 * ManageReferencesModal — lists every user-uploaded reference clip with
 * its name/description and a delete trashcan per row. Opened from the
 * Design tab's "Manage" button next to the Base voice dropdown.
 *
 * Delete is immediate (no staging); the DELETE API call fires per row
 * and the list updates in place. Previously-saved voices that used a
 * deleted reference keep working — their own audio lives under
 * data/profiles/, independent of data/presets/user/.
 */

import { useState } from "react";
import { deleteReference, type Reference } from "@/lib/api";

type Props = {
  open: boolean;
  onClose: () => void;
  references: Reference[];
  onChange: (next: Reference[]) => void;
};

export function ManageReferencesModal({
  open,
  onClose,
  references,
  onChange,
}: Props) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  async function handleDelete(ref: Reference) {
    if (
      !confirm(
        `Delete "${ref.name}"?\n\nRemoves the reference clip from your local machine. Voices you already saved using it are unaffected.`
      )
    ) {
      return;
    }
    setBusyId(ref.id);
    setError(null);
    try {
      const slug = ref.id.startsWith("user:") ? ref.id.slice(5) : ref.id;
      await deleteReference(slug);
      onChange(references.filter((r) => r.id !== ref.id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyId(null);
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
        className="card w-full max-w-md m-4 max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold">Manage uploaded voices</h2>
          <button
            onClick={onClose}
            className="text-[color:var(--muted)] hover:text-[color:var(--foreground)] text-lg"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {references.length === 0 ? (
          <div className="text-sm text-[color:var(--muted)] py-6 text-center">
            No uploaded references yet. Use the ⇡ Upload button on the
            Design tab to add one.
          </div>
        ) : (
          <ul className="flex flex-col gap-2 overflow-y-auto pr-1 min-h-0">
            {references.map((r) => (
              <li
                key={r.id}
                className="flex items-start justify-between gap-3 p-2.5 rounded-lg border border-[color:var(--border)]"
              >
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-sm truncate">{r.name}</div>
                  {r.description && (
                    <div className="text-xs text-[color:var(--muted)] mt-0.5 leading-snug">
                      {r.description}
                    </div>
                  )}
                </div>
                <button
                  className="btn btn-ghost !p-1.5 flex-shrink-0"
                  onClick={() => handleDelete(r)}
                  disabled={busyId === r.id}
                  title="Delete this reference"
                  aria-label={`Delete ${r.name}`}
                >
                  <TrashIcon />
                </button>
              </li>
            ))}
          </ul>
        )}

        {error && (
          <div className="text-xs text-red-500 mt-3 break-all">{error}</div>
        )}

        <div className="flex justify-end mt-4">
          <button className="btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

function TrashIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="text-[color:var(--muted)] hover:text-red-500 transition-colors"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}
