/**
 * Interactive transcript editor with speaker renaming and re-summarization.
 *
 * Supports: inline text editing, speaker reassignment, speaker renaming
 * (click speaker name to rename globally), save edits, and re-summarize
 * to regenerate the protocol with updated transcript.
 */

import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Save,
  ArrowRight,
  Loader2,
  AlertCircle,
  Check,
  Edit3,
  User,
  Clock,
  RefreshCw,
  Users,
  BookOpen,
  BadgeCheck,
  GitMerge,
  Plus,
  Scissors,
  XCircle,
} from "lucide-react";
import {
  getTranscript,
  getTranscriptReview,
  confirmSpeakerReview,
  saveTranscript,
  listSpeakers,
  renameSpeaker,
  resummarize,
} from "../api/client";
import type {
  AlignedSegment,
  SpeakerProfile,
  TranscriptReview,
} from "../types/api";

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Tailwind classes for the segment card based on attribution_confidence.
 * Low values = contested attribution (overlap region, brief turn,
 * multi-speaker fight) and need editor review. Thresholds match the
 * confidence buckets the aligner emits:
 *   < 0.6  red   "contested"
 *   < 0.8  amber "marginal"
 *   >= 0.8 default (no extra styling)
 *   null   default (no diarization to score against)
 */
function attributionStyle(conf: number | null): string {
  if (conf === null || conf === undefined) {
    return "border-epam-gray-200";
  }
  if (conf < 0.6) {
    return "border-red-300 bg-red-50/40";
  }
  if (conf < 0.8) {
    return "border-amber-300 bg-amber-50/40";
  }
  return "border-epam-gray-200";
}

/** Deterministic color for a speaker_id. */
function speakerColor(speakerId: string): string {
  const colors = [
    "bg-blue-100 text-blue-700",
    "bg-emerald-100 text-emerald-700",
    "bg-purple-100 text-purple-700",
    "bg-amber-100 text-amber-700",
    "bg-rose-100 text-rose-700",
    "bg-cyan-100 text-cyan-700",
    "bg-orange-100 text-orange-700",
    "bg-indigo-100 text-indigo-700",
    "bg-teal-100 text-teal-700",
    "bg-pink-100 text-pink-700",
  ];
  let hash = 0;
  for (let i = 0; i < speakerId.length; i++) {
    hash = (hash * 31 + speakerId.charCodeAt(i)) | 0;
  }
  return colors[Math.abs(hash) % colors.length];
}

export default function TranscriptPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [segments, setSegments] = useState<AlignedSegment[]>([]);
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [review, setReview] = useState<TranscriptReview | null>(null);
  const [confirmingReview, setConfirmingReview] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editIdx, setEditIdx] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [dirty, setDirty] = useState(false);

  // Speaker rename state
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameLoading, setRenameLoading] = useState(false);

  // Mass rename state (Sprint 2026-04-30, task #41).
  // Opens a modal listing every detected speaker with sample turns
  // and lets the user fill in real names for all of them in one
  // form, then commit each via renameSpeaker. Cuts the
  // click-rename-click-rename loop on big admin meetings.
  const [massRenameOpen, setMassRenameOpen] = useState(false);
  const [massRenameMap, setMassRenameMap] = useState<Record<string, string>>({});
  const [massRenameSaving, setMassRenameSaving] = useState(false);

  // Re-summarize state
  const [resummarizing, setResummarizing] = useState(false);

  // Tier 2 (2026-04-23): low-attribution-confidence filter. When true,
  // hide segments whose attribution_confidence is null or >= 0.8 — i.e.
  // show only the ones the editor needs to verify (overlap regions,
  // contested speakers, brief turns). Aligner downgrades confidence for
  // overlap regions to <=0.55 so they always land in this view.
  const [reviewOnly, setReviewOnly] = useState(false);
  const [manualSpeakers, setManualSpeakers] = useState<string[]>([]);

  useEffect(() => {
    if (!jobId) return;

    async function load() {
      try {
        setReview(null);
        const [segs, spkrs] = await Promise.all([
          getTranscript(jobId!),
          listSpeakers().catch(() => []),
        ]);
        setSegments(segs);
        setSpeakers(spkrs);
        getTranscriptReview(jobId!)
          .then(setReview)
          .catch(() => setReview(null));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load transcript");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [jobId]);

  const speakerNameMap = new Map<string, string>();
  for (const s of speakers) {
    speakerNameMap.set(s.id, s.name);
  }

  // Collect unique speakers from transcript
  const uniqueSpeakers = Array.from(
    new Set([...segments.map((s) => s.speaker_id), ...manualSpeakers]),
  );

  function addManualSpeaker() {
    const existing = new Set(uniqueSpeakers);
    let idx = uniqueSpeakers.length + 1;
    let id = `MANUAL_${String(idx).padStart(2, "0")}`;
    while (existing.has(id)) {
      idx += 1;
      id = `MANUAL_${String(idx).padStart(2, "0")}`;
    }
    setManualSpeakers((prev) => [...prev, id]);
    setDirty(true);
  }

  function startEdit(idx: number) {
    setEditIdx(idx);
    setEditText(segments[idx].text);
  }

  function cancelEdit() {
    setEditIdx(null);
    setEditText("");
  }

  function confirmEdit() {
    if (editIdx === null) return;
    const updated = [...segments];
    updated[editIdx] = { ...updated[editIdx], text: editText };
    setSegments(updated);
    setDirty(true);
    setEditIdx(null);
    setEditText("");
  }

  function changeSpeaker(idx: number, newSpeakerId: string) {
    const updated = [...segments];
    updated[idx] = {
      ...updated[idx],
      speaker_id: newSpeakerId,
      speaker_name: speakerNameMap.get(newSpeakerId) || null,
    };
    setSegments(updated);
    setDirty(true);
  }

  function acceptSuggestion(idx: number) {
    const seg = segments[idx];
    const suggestion = seg?.suggested_speaker_name?.trim();
    if (!seg || !suggestion) return;
    setSegments((prev) =>
      prev.map((item) =>
        item.speaker_id === seg.speaker_id
          ? { ...item, speaker_name: suggestion }
          : item,
      ),
    );
    setDirty(true);
  }

  function rejectSuggestion(idx: number) {
    const seg = segments[idx];
    if (!seg) return;
    setSegments((prev) =>
      prev.map((item) =>
        item.speaker_id === seg.speaker_id
          ? {
              ...item,
              suggested_speaker_name: null,
              suggested_speaker_confidence: null,
            }
          : item,
      ),
    );
    setDirty(true);
  }

  function findSplitOffset(text: string): number {
    const middle = Math.floor(text.length / 2);
    const windowSize = Math.max(12, Math.floor(text.length * 0.25));
    const from = Math.max(1, middle - windowSize);
    const to = Math.min(text.length - 1, middle + windowSize);
    const candidates = [". ", "? ", "! ", "; ", ", ", " "];
    for (const marker of candidates) {
      let best = -1;
      for (let i = from; i < to; i++) {
        if (text.slice(i, i + marker.length) === marker) {
          if (best < 0 || Math.abs(i - middle) < Math.abs(best - middle)) {
            best = i + marker.length;
          }
        }
      }
      if (best > 0) return best;
    }
    return middle;
  }

  function splitSegment(idx: number) {
    const seg = segments[idx];
    if (!seg) return;
    const text = (seg.text || "").trim();
    if (text.length < 8) return;
    const cut = findSplitOffset(text);
    const left = text.slice(0, cut).trim();
    const right = text.slice(cut).trim();
    if (!left || !right) return;
    const duration = Math.max(0.02, seg.end - seg.start);
    const ratio = left.length / (left.length + right.length);
    const splitAt = seg.start + duration * ratio;
    const updated = [...segments];
    updated.splice(
      idx,
      1,
      { ...seg, end: splitAt, text: left },
      { ...seg, start: splitAt, text: right },
    );
    setSegments(updated);
    setDirty(true);
  }

  function mergeSegment(idx: number, direction: -1 | 1) {
    const neighborIdx = idx + direction;
    if (neighborIdx < 0 || neighborIdx >= segments.length) return;
    const firstIdx = Math.min(idx, neighborIdx);
    const secondIdx = Math.max(idx, neighborIdx);
    const first = segments[firstIdx];
    const second = segments[secondIdx];
    const merged: AlignedSegment = {
      ...first,
      end: Math.max(first.end, second.end),
      text: `${(first.text || "").trim()} ${(second.text || "").trim()}`.trim(),
      confidence: Math.min(first.confidence ?? 1, second.confidence ?? 1),
      attribution_confidence:
        first.attribution_confidence === null ||
        first.attribution_confidence === undefined ||
        second.attribution_confidence === null ||
        second.attribution_confidence === undefined
          ? null
          : Math.min(first.attribution_confidence, second.attribution_confidence),
    };
    const updated = [...segments];
    updated.splice(firstIdx, 2, merged);
    setSegments(updated);
    setDirty(true);
  }

  // Speaker rename: click speaker badge in legend to rename globally
  function startRename(speakerId: string) {
    const current = segments.find((s) => s.speaker_id === speakerId);
    setRenamingId(speakerId);
    setRenameValue(current?.speaker_name || speakerId);
  }

  async function confirmRename() {
    if (!renamingId || !jobId || !renameValue.trim()) return;
    setRenameLoading(true);
    setError(null);
    try {
      const oldName = renamingId;
      await renameSpeaker(jobId, oldName, renameValue.trim());
      // Update local segments
      setSegments((prev) =>
        prev.map((seg) =>
          seg.speaker_id === oldName
            ? { ...seg, speaker_name: renameValue.trim() }
            : seg,
        ),
      );
      setRenamingId(null);
      setRenameValue("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rename failed");
    } finally {
      setRenameLoading(false);
    }
  }

  // Mass rename: open modal pre-populated with current display names.
  function openMassRename() {
    const initial: Record<string, string> = {};
    for (const sid of uniqueSpeakers) {
      const seg = segments.find((s) => s.speaker_id === sid);
      initial[sid] =
        seg?.speaker_name || seg?.suggested_speaker_name || sid;
    }
    setMassRenameMap(initial);
    setMassRenameOpen(true);
  }

  // Sample 2-3 short turns per speaker so the user can recognise
  // who it is without clicking around the transcript.
  function sampleTurnsFor(speakerId: string, max = 3): string[] {
    const out: string[] = [];
    for (const s of segments) {
      if (s.speaker_id !== speakerId) continue;
      const t = (s.text || "").trim();
      if (!t) continue;
      out.push(t.length > 90 ? t.slice(0, 87) + "..." : t);
      if (out.length >= max) break;
    }
    return out;
  }

  async function commitMassRename() {
    if (!jobId) return;
    setMassRenameSaving(true);
    setError(null);
    try {
      const successfulRenames: Array<[string, string]> = [];
      for (const [oldName, newName] of Object.entries(massRenameMap)) {
        const trimmed = newName.trim();
        if (!trimmed || trimmed === oldName) continue;
        await renameSpeaker(jobId, oldName, trimmed);
        successfulRenames.push([oldName, trimmed]);
      }
      // Apply to local state.
      setSegments((prev) =>
        prev.map((seg) => {
          const target = successfulRenames.find(([o]) => o === seg.speaker_id);
          if (target) {
            return { ...seg, speaker_name: target[1] };
          }
          return seg;
        }),
      );
      setMassRenameOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mass rename failed");
    } finally {
      setMassRenameSaving(false);
    }
  }

  const handleSave = useCallback(async () => {
    if (!jobId) return;
    setSaving(true);
    setSaved(false);
    setError(null);

    try {
      await saveTranscript(jobId, segments);
      setDirty(false);
      setSaved(true);
      getTranscriptReview(jobId).then(setReview).catch(() => undefined);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }, [jobId, segments]);

  async function handleResummarize() {
    if (!jobId) return;
    setResummarizing(true);
    setError(null);

    try {
      // Save first if dirty
      if (dirty) {
        await saveTranscript(jobId, segments);
        setDirty(false);
      }
      await resummarize(jobId);
      navigate(`/processing/${jobId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Re-summarization failed");
      setResummarizing(false);
    }
  }

  async function handleConfirmSpeakerReview() {
    if (!jobId) return;
    setConfirmingReview(true);
    setError(null);
    try {
      if (dirty) {
        await saveTranscript(jobId, segments);
        setDirty(false);
      }
      await confirmSpeakerReview(jobId);
      const updated = await getTranscriptReview(jobId);
      setReview(updated);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Speaker review confirmation failed",
      );
    } finally {
      setConfirmingReview(false);
    }
  }

  // Ctrl+S to save
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (dirty && !saving) handleSave();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [dirty, saving, handleSave]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={24} className="animate-spin text-epam-red" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-heading text-2xl text-epam-gray-900">
            Transcript Editor
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            {segments.length} segments &middot; {uniqueSpeakers.length} speakers
            &middot; Click speaker names below to rename
          </p>
        </div>

        <div className="flex items-center gap-3">
          {saved && (
            <span className="flex items-center gap-1 text-sm text-green-600">
              <Check size={16} /> Saved
            </span>
          )}
          <button
            onClick={handleSave}
            disabled={saving || !dirty}
            className="flex items-center gap-2 px-4 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Save size={14} />
            )}
            Save
          </button>
          <button
            onClick={handleResummarize}
            disabled={resummarizing}
            className="flex items-center gap-2 px-4 py-2 border border-epam-red text-epam-red rounded-lg text-sm font-medium hover:bg-epam-red/5 transition-colors disabled:opacity-50"
          >
            {resummarizing ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <RefreshCw size={14} />
            )}
            Re-summarize
          </button>
          <button
            onClick={() => navigate(`/protocol/${jobId}`)}
            className="flex items-center gap-2 px-4 py-2 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors"
          >
            Protocol
            <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {review && (
        <div
          className={`mb-4 border rounded-lg p-4 text-sm ${
            review.speaker_count_matches === false
              ? "border-red-200 bg-red-50/50"
              : "border-epam-gray-200 bg-white"
          }`}
        >
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <div className="flex items-center gap-2">
              <Users size={16} className="text-epam-red" />
              <span className="font-medium text-epam-gray-800">
                Speakers: {review.detected_speakers}
              </span>
              {review.expected_speakers_min !== null &&
                review.expected_speakers_max !== null && (
                  <span
                    className={
                      review.speaker_count_matches === false
                        ? "text-red-700"
                        : "text-epam-gray-500"
                    }
                  >
                    expected{" "}
                    {review.expected_speakers_min ===
                    review.expected_speakers_max
                      ? review.expected_speakers_min
                      : `${review.expected_speakers_min}-${review.expected_speakers_max}`}
                  </span>
                )}
            </div>
            <div className="flex items-center gap-2 text-epam-gray-600">
              <BookOpen size={16} className="text-epam-gray-400" />
              <span>
                Dictionary:{" "}
                {review.dictionary_terms.length -
                  review.dictionary_missing_count}
                /{review.dictionary_terms.length} found
              </span>
            </div>
            <div className="text-epam-gray-600">
              {review.low_confidence_turns} turns need attribution review
            </div>
          </div>

          {review.speaker_count_matches === false && (
            <p className="mt-2 text-xs text-red-700">
              Detected speaker count does not match the upload setting. Use
              mass rename and the review-only filter before final DOCX export.
            </p>
          )}

          {review.meeting_type === "court_hearing" &&
            review.expected_speakers_min !== null &&
            review.expected_speakers_min === review.expected_speakers_max &&
            review.detected_speakers < review.expected_speakers_min && (
              <button
                type="button"
                onClick={addManualSpeaker}
                className="mt-3 inline-flex items-center gap-1.5 rounded border border-epam-red px-3 py-1.5 text-xs font-medium text-epam-red hover:bg-epam-red/5"
              >
                <Plus size={13} />
                Add missing speaker
              </button>
            )}

          {review.dictionary_missing_count > 0 && (
            <div className="mt-2 text-xs text-epam-gray-600">
              <span className="font-medium">Missing dictionary terms: </span>
              {review.dictionary_terms
                .filter((t) => !t.found)
                .slice(0, 8)
                .map((t) => t.term)
                .join(", ")}
              {review.dictionary_missing_count > 8 ? " ..." : ""}
            </div>
          )}

          {review.speakers.some((s) => s.suggested_speaker_name) && (
            <div className="mt-2 text-xs text-epam-gray-600">
              Voice-profile suggestions are pre-filled in mass rename, but
              court hearings still require human confirmation.
            </div>
          )}

          {review.meeting_type === "court_hearing" && (
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-epam-gray-200 pt-3">
              <div className="text-xs text-epam-gray-600">
                {review.speaker_review?.confirmed ? (
                  <span className="text-green-700 font-medium">
                    Speakers confirmed by {review.speaker_review.confirmed_by}
                  </span>
                ) : (
                  <span>
                    Final DOCX download is locked until speakers are confirmed.
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={handleConfirmSpeakerReview}
                disabled={confirmingReview || review.speaker_review?.confirmed}
                className="px-3 py-1.5 bg-epam-red text-white rounded text-xs font-medium hover:bg-epam-red-dark disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {confirmingReview
                  ? "Confirming..."
                  : review.speaker_review?.confirmed
                    ? "Confirmed"
                    : "Confirm speakers"}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Review-only filter + low-confidence stats */}
      {(() => {
        const lowCount = segments.filter(
          (s) =>
            s.attribution_confidence !== null &&
            s.attribution_confidence !== undefined &&
            s.attribution_confidence < 0.8,
        ).length;
        if (segments.length === 0) return null;
        return (
          <div className="mb-4 flex items-center justify-between text-xs">
            <div className="text-epam-gray-500">
              {lowCount > 0 ? (
                <>
                  <span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />
                  <span className="font-medium text-epam-gray-700">{lowCount}</span>{" "}
                  из {segments.length} реплик помечены к проверке
                  (наложение голосов или спорная атрибуция).
                </>
              ) : (
                <>Все {segments.length} реплик с уверенной атрибуцией.</>
              )}
            </div>
            {lowCount > 0 && (
              <button
                onClick={() => setReviewOnly((v) => !v)}
                className={`px-2 py-1 rounded font-medium border transition-colors ${
                  reviewOnly
                    ? "bg-amber-50 border-amber-300 text-amber-700"
                    : "border-epam-gray-300 text-epam-gray-600 hover:border-epam-gray-400"
                }`}
              >
                {reviewOnly ? "Показать все" : "Только к проверке"}
              </button>
            )}
          </div>
        );
      })()}

      {/* Speaker legend — click to rename, or open mass-rename modal */}
      <div className="mb-6 flex flex-wrap items-center gap-2">
        {uniqueSpeakers.length > 1 && (
          <button
            onClick={openMassRename}
            className="flex items-center gap-1 px-3 py-1 border border-epam-gray-300 text-epam-gray-700 rounded text-xs font-medium hover:bg-epam-gray-100 transition-colors mr-2"
            title="Переименовать всех спикеров за раз"
          >
            <Edit3 size={12} />
            Переименовать всех
          </button>
        )}
        <button
          onClick={addManualSpeaker}
          className="flex items-center gap-1 px-3 py-1 border border-epam-gray-300 text-epam-gray-700 rounded text-xs font-medium hover:bg-epam-gray-100 transition-colors mr-2"
          title="Add a manual speaker ID for reassigning turns"
        >
          <Plus size={12} />
          Add speaker
        </button>
        {uniqueSpeakers.map((sid) => {
          const name = segments.find((s) => s.speaker_id === sid)?.speaker_name
            || speakerNameMap.get(sid) || sid;

          if (renamingId === sid) {
            return (
              <div key={sid} className="flex items-center gap-1">
                <input
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") confirmRename();
                    if (e.key === "Escape") setRenamingId(null);
                  }}
                  autoFocus
                  className="px-2 py-1 border border-epam-red rounded text-xs w-36 focus:outline-none focus:ring-1 focus:ring-epam-red"
                />
                <button
                  onClick={confirmRename}
                  disabled={renameLoading}
                  className="px-2 py-1 bg-epam-red text-white text-xs rounded font-medium hover:bg-epam-red-dark disabled:opacity-50"
                >
                  {renameLoading ? "..." : "OK"}
                </button>
                <button
                  onClick={() => setRenamingId(null)}
                  className="px-2 py-1 text-xs text-epam-gray-500 hover:text-epam-gray-700"
                >
                  Cancel
                </button>
              </div>
            );
          }

          return (
            <button
              key={sid}
              onClick={() => startRename(sid)}
              className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium cursor-pointer hover:ring-2 hover:ring-epam-red/30 transition-all ${speakerColor(sid)}`}
              title={`Click to rename ${name}`}
            >
              <User size={12} />
              {name}
              <Edit3 size={10} className="opacity-50" />
            </button>
          );
        })}
      </div>

      {/* Transcript segments */}
      <div className="space-y-2">
        {segments.map((seg, idx) => {
          // T2.3 (2026-04-23): in review-only mode, hide segments
          // that have no confidence problem. Keep their original
          // index for changeSpeaker / startEdit.
          if (
            reviewOnly &&
            seg.attribution_confidence !== null &&
            seg.attribution_confidence !== undefined &&
            seg.attribution_confidence >= 0.8
          ) {
            return null;
          }
          const attribStyle = attributionStyle(
            seg.attribution_confidence ?? null,
          );
          return (
          <div
            key={idx}
            className={`group flex gap-3 p-3 rounded-lg bg-white border hover:border-epam-gray-400 transition-colors ${attribStyle}`}
          >
            {/* Timestamp */}
            <div className="shrink-0 pt-0.5">
              <span className="flex items-center gap-1 text-xs text-epam-gray-400 font-mono whitespace-nowrap">
                <Clock size={11} />
                {formatTime(seg.start)}
              </span>
            </div>

            {/* Speaker badge */}
            <div className="shrink-0 pt-0.5">
              <select
                value={seg.speaker_id}
                onChange={(e) => changeSpeaker(idx, e.target.value)}
                className={`text-xs font-medium rounded px-2 py-1 border-0 cursor-pointer ${speakerColor(seg.speaker_id)}`}
              >
                {uniqueSpeakers.map((sid) => (
                  <option key={sid} value={sid}>
                    {segments.find((s) => s.speaker_id === sid)?.speaker_name || speakerNameMap.get(sid) || sid}
                  </option>
                ))}
              </select>
            </div>

            {/* Text */}
            <div className="flex-1 min-w-0">
              {editIdx === idx ? (
                <div className="space-y-2">
                  <textarea
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    rows={3}
                    className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red"
                    autoFocus
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={confirmEdit}
                      className="px-3 py-1 bg-epam-red text-white text-xs rounded font-medium hover:bg-epam-red-dark"
                    >
                      Apply
                    </button>
                    <button
                      onClick={cancelEdit}
                      className="px-3 py-1 border border-epam-gray-300 text-epam-gray-600 text-xs rounded font-medium hover:bg-epam-gray-100"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p
                    className="text-sm text-epam-gray-800 leading-relaxed cursor-text"
                    onClick={() => startEdit(idx)}
                  >
                    {seg.text}
                    <Edit3
                      size={12}
                      className="inline ml-2 text-epam-gray-300 group-hover:text-epam-gray-400"
                    />
                  </p>
                  {seg.suggested_speaker_name &&
                    seg.suggested_speaker_name !== seg.speaker_name && (
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-epam-gray-600">
                        <span>
                          Voice suggestion:{" "}
                          <strong>{seg.suggested_speaker_name}</strong>
                          {seg.suggested_speaker_confidence !== null &&
                            seg.suggested_speaker_confidence !== undefined && (
                              <span className="text-epam-gray-400">
                                {" "}
                                ({Math.round(seg.suggested_speaker_confidence * 100)}%)
                              </span>
                            )}
                        </span>
                        <button
                          type="button"
                          onClick={() => acceptSuggestion(idx)}
                          className="inline-flex items-center gap-1 rounded border border-green-200 bg-green-50 px-2 py-0.5 font-medium text-green-700 hover:bg-green-100"
                        >
                          <BadgeCheck size={12} />
                          Accept
                        </button>
                        <button
                          type="button"
                          onClick={() => rejectSuggestion(idx)}
                          className="inline-flex items-center gap-1 rounded border border-epam-gray-200 px-2 py-0.5 font-medium text-epam-gray-600 hover:bg-epam-gray-100"
                        >
                          <XCircle size={12} />
                          Reject
                        </button>
                      </div>
                    )}
                </>
              )}
            </div>

            <div className="shrink-0 pt-0.5 flex items-center gap-1 opacity-70 group-hover:opacity-100">
              <button
                type="button"
                onClick={() => splitSegment(idx)}
                className="rounded p-1 text-epam-gray-400 hover:bg-epam-gray-100 hover:text-epam-gray-700 disabled:opacity-30"
                title="Split this turn"
                disabled={(seg.text || "").trim().length < 8}
              >
                <Scissors size={14} />
              </button>
              <button
                type="button"
                onClick={() => mergeSegment(idx, -1)}
                className="rounded p-1 text-epam-gray-400 hover:bg-epam-gray-100 hover:text-epam-gray-700 disabled:opacity-30"
                title="Merge with previous turn"
                disabled={idx === 0}
              >
                <GitMerge size={14} />
              </button>
              <button
                type="button"
                onClick={() => mergeSegment(idx, 1)}
                className="rounded p-1 text-epam-gray-400 hover:bg-epam-gray-100 hover:text-epam-gray-700 disabled:opacity-30"
                title="Merge with next turn"
                disabled={idx >= segments.length - 1}
              >
                <GitMerge size={14} className="rotate-180" />
              </button>
            </div>

            {/* Confidence */}
            {seg.confidence < 0.8 && (
              <div className="shrink-0 pt-0.5">
                <span
                  className="text-xs text-amber-500"
                  title={`Confidence: ${(seg.confidence * 100).toFixed(0)}%`}
                >
                  {(seg.confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
            {/* Attribution confidence badge for low-conf segments */}
            {seg.attribution_confidence !== null &&
              seg.attribution_confidence !== undefined &&
              seg.attribution_confidence < 0.8 && (
                <div className="shrink-0 pt-0.5">
                  <span
                    className={`text-[10px] font-medium px-1.5 py-0.5 rounded uppercase tracking-wide ${
                      seg.attribution_confidence < 0.6
                        ? "bg-red-100 text-red-700"
                        : "bg-amber-100 text-amber-700"
                    }`}
                    title={`Attribution confidence: ${(
                      seg.attribution_confidence * 100
                    ).toFixed(0)}% — наложение голосов или спорная атрибуция, проверьте говорящего`}
                  >
                    {seg.attribution_confidence < 0.6 ? "проверка" : "спорно"}
                  </span>
                </div>
              )}
          </div>
          );
        })}
      </div>

      {segments.length === 0 && !error && (
        <div className="text-center py-16 text-epam-gray-400 text-sm">
          No transcript segments found. The meeting may still be processing.
        </div>
      )}

      {/* Mass rename modal (Sprint 2026-04-30, task #41) */}
      {massRenameOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 max-w-2xl w-full max-h-[85vh] overflow-y-auto">
            <h3 className="font-heading text-lg text-epam-gray-900 mb-2">
              Переименование спикеров
            </h3>
            <p className="text-xs text-epam-gray-500 mb-4">
              Образцы реплик помогут опознать каждого. Пустые поля
              оставят текущее имя без изменений.
            </p>
            <div className="space-y-4">
              {uniqueSpeakers.map((sid) => {
                const samples = sampleTurnsFor(sid);
                const speakerSeg = segments.find((s) => s.speaker_id === sid);
                const voiceSuggestion = speakerSeg?.suggested_speaker_name;
                const voiceConfidence =
                  speakerSeg?.suggested_speaker_confidence;
                const suggestion = speakers.find(
                  (s) =>
                    s.name &&
                    massRenameMap[sid] &&
                    s.name.toLowerCase() ===
                      (massRenameMap[sid] || "").toLowerCase(),
                );
                return (
                  <div
                    key={sid}
                    className="border border-epam-gray-200 rounded-lg p-3"
                  >
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <span className="text-xs font-mono text-epam-gray-500 shrink-0">
                        {sid}
                      </span>
                      <input
                        type="text"
                        value={massRenameMap[sid] || ""}
                        onChange={(e) =>
                          setMassRenameMap((prev) => ({
                            ...prev,
                            [sid]: e.target.value,
                          }))
                        }
                        placeholder="Новое имя"
                        className="flex-1 px-3 py-1.5 border border-epam-gray-300 rounded text-sm focus:outline-none focus:border-epam-red"
                      />
                    </div>
                    {voiceSuggestion && (
                      <div className="mb-2 text-xs text-epam-gray-600">
                        Voice profile suggests:{" "}
                        <button
                          type="button"
                          onClick={() =>
                            setMassRenameMap((prev) => ({
                              ...prev,
                              [sid]: voiceSuggestion,
                            }))
                          }
                          className="font-medium text-epam-red hover:underline"
                        >
                          {voiceSuggestion}
                        </button>
                        {voiceConfidence !== null &&
                          voiceConfidence !== undefined && (
                            <span className="text-epam-gray-400">
                              {" "}
                              ({Math.round(voiceConfidence * 100)}%)
                            </span>
                          )}
                      </div>
                    )}
                    {speakers.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-2">
                        {speakers.slice(0, 8).map((p) => (
                          <button
                            key={p.id}
                            type="button"
                            onClick={() =>
                              setMassRenameMap((prev) => ({
                                ...prev,
                                [sid]: p.name,
                              }))
                            }
                            className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                              suggestion?.id === p.id
                                ? "border-epam-red text-epam-red bg-epam-red/5"
                                : "border-epam-gray-200 text-epam-gray-600 hover:border-epam-gray-400"
                            }`}
                          >
                            {p.name}
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="space-y-1 text-xs text-epam-gray-600">
                      {samples.length === 0 ? (
                        <em>Нет реплик от этого спикера.</em>
                      ) : (
                        samples.map((t, i) => (
                          <p
                            key={i}
                            className="border-l-2 border-epam-gray-200 pl-2"
                          >
                            {t}
                          </p>
                        ))
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-end gap-2 mt-5">
              <button
                onClick={() => setMassRenameOpen(false)}
                disabled={massRenameSaving}
                className="px-4 py-2 text-sm text-epam-gray-600 hover:text-epam-gray-900"
              >
                Отмена
              </button>
              <button
                onClick={commitMassRename}
                disabled={massRenameSaving}
                className="px-4 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark disabled:opacity-50"
              >
                {massRenameSaving ? "Сохранение..." : "Применить ко всем"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
