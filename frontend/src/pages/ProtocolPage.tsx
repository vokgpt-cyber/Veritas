/**
 * Protocol viewer — shows meeting summary, participants, transcript,
 * decisions/tasks/risks, and download buttons.
 *
 * Dispatches on `meeting_type`:
 *   - "administrative" → AdminProtocolView (T3.3 flat 9-section schema)
 *   - "court_hearing"  → CourtProtocolView  (verbatim stenogram + summary)
 *   - "client_meeting" → ClientProtocolView (commitments + follow-ups)
 *   - "interview"      → InterviewProtocolView (internal + external reports)
 *   - else / legacy    → LegacyProtocolView   (pre-T3.3 MeetingProtocol)
 *
 * Backend's GET /protocol/data returns either the legacy `MeetingProtocol`
 * (flat fields) OR a wrapped `{meeting_type, payload}` envelope. The
 * `isWrappedProtocol` type guard discriminates them and lets each
 * sub-view consume its own typed payload without `any` gymnastics.
 *
 * Stats row (top-right card) is computed per meeting type from whichever
 * fields are populated — admin/court use `turns[]`, legacy uses
 * `transcript[]`. No more 0/0/0:00 — that was the bug fixed 2026-04-30
 * when the wrapped admin payload landed but ProtocolPage hadn't been
 * updated to read it.
 */

import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Download,
  FileText,
  Loader2,
  AlertCircle,
  Cloud,
  Users,
  MessageSquare,
  ChevronLeft,
  Edit3,
  CheckCircle,
  Clock,
  AlertTriangle,
  HelpCircle,
} from "lucide-react";
import {
  getMeetingStatus,
  getProtocolData,
  getProtocolUrl,
  getSystemFeatures,
  getToken,
  summarizeCloud,
} from "../api/client";
import type {
  MeetingJob,
  MeetingProtocol,
  WrappedProtocol,
  AdministrativePayload,
  CourtHearingPayload,
  ClientMeetingPayload,
  InterviewPayload,
  StenogramTurn,
  FlatProtocolItem,
} from "../types/api";
import { isWrappedProtocol } from "../types/api";

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0)
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

/** Compute (speakers, segments, duration_s) for the stats row. */
function computeStats(
  data: MeetingProtocol | WrappedProtocol | null,
): { speakers: number; segments: number; duration: number } {
  if (!data) return { speakers: 0, segments: 0, duration: 0 };
  if (isWrappedProtocol(data)) {
    const mt = data.meeting_type;
    if (mt === "administrative" || mt === "court_hearing") {
      const p = data.payload as AdministrativePayload | CourtHearingPayload;
      const turns = p.turns || [];
      // Distinct speakers in the stenogram (more reliable than the
      // separate `participants` array which may be a static attendee
      // list rather than detected speakers).
      const speakerSet = new Set<string>();
      for (const t of turns) if (t.speaker) speakerSet.add(t.speaker);
      // Duration: from first turn's start to last turn's start.
      // We don't have explicit end times on turns — start-to-start is
      // a close-enough proxy and avoids showing 0:00 when present.
      let duration = 0;
      const withTime = turns.filter((t) => typeof t.start_s === "number");
      if (withTime.length >= 2) {
        const first = withTime[0].start_s as number;
        const last = withTime[withTime.length - 1].start_s as number;
        duration = Math.max(0, last - first);
      }
      return {
        speakers: speakerSet.size || (p.participants || []).length,
        segments: turns.length,
        duration,
      };
    }
    if (mt === "client_meeting") {
      const p = data.payload as ClientMeetingPayload;
      const speakers =
        (p.epam_representatives || []).length +
        (p.client_representatives || []).length;
      // No stenogram on client meetings — count meaningful items instead.
      const segments =
        (p.topics || []).length +
        (p.commitments || []).length +
        (p.follow_ups || []).length;
      return { speakers, segments, duration: 0 };
    }
    // interview: not really applicable, but show something non-zero.
    return { speakers: 0, segments: 0, duration: 0 };
  }
  // Legacy MeetingProtocol shape.
  const legacy = data as MeetingProtocol;
  const transcript = legacy.transcript || [];
  let duration = 0;
  if (transcript.length > 0) {
    duration =
      transcript[transcript.length - 1].end - transcript[0].start;
  }
  return {
    speakers: (legacy.participants || []).length,
    segments: transcript.length,
    duration,
  };
}

export default function ProtocolPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [job, setJob] = useState<MeetingJob | null>(null);
  const [protocol, setProtocol] = useState<
    MeetingProtocol | WrappedProtocol | null
  >(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Claude cloud comparison
  const [cloudSummary, setCloudSummary] = useState<string | null>(null);
  const [cloudLoading, setCloudLoading] = useState(false);
  const [cloudMeta, setCloudMeta] = useState<{
    model: string;
    tokens_in: number;
    tokens_out: number;
  } | null>(null);
  const [cloudCompareEnabled, setCloudCompareEnabled] = useState(false);

  async function handleCloudSummarize() {
    if (!jobId) return;
    setCloudLoading(true);
    setError(null);
    try {
      const result = await summarizeCloud(jobId);
      setCloudSummary(result.summary);
      setCloudMeta({
        model: result.model,
        tokens_in: result.tokens_in,
        tokens_out: result.tokens_out,
      });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Cloud summarization failed",
      );
    } finally {
      setCloudLoading(false);
    }
  }

  useEffect(() => {
    if (!jobId) return;

    async function load() {
      try {
        const [jobData, proto, features] = await Promise.all([
          getMeetingStatus(jobId!),
          getProtocolData(jobId!).catch(() => null),
          getSystemFeatures().catch(() => null),
        ]);
        setJob(jobData);
        setProtocol(proto);
        setCloudCompareEnabled(
          Boolean(features?.cloud_compare?.default_enabled),
        );
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load protocol",
        );
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [jobId]);

  /** Download protocol file with JWT auth header. */
  async function downloadProtocol(id: string, format: string) {
    try {
      const url = getProtocolUrl(id, format);
      const token = getToken();
      const response = await fetch(url, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Download failed: ${response.status}`);
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `protocol.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={24} className="animate-spin text-epam-red" />
      </div>
    );
  }

  const stats = computeStats(protocol);

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-heading text-2xl text-epam-gray-900">
            Meeting Protocol
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            {job?.filename ?? "Meeting"} &middot;{" "}
            {job?.created_at
              ? new Date(job.created_at).toLocaleDateString("ru-RU")
              : ""}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(`/transcript/${jobId}`)}
            className="flex items-center gap-2 px-4 py-2 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors"
          >
            <Edit3 size={14} />
            Edit Transcript
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Download buttons */}
      {jobId && (
        <div className="mb-8 flex flex-wrap gap-3">
          <button
            onClick={() => downloadProtocol(jobId, "docx")}
            className="flex items-center gap-2 px-5 py-2.5 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors"
          >
            <Download size={16} />
            Download DOCX
          </button>
          <button
            onClick={() => downloadProtocol(jobId, "json")}
            className="flex items-center gap-2 px-5 py-2.5 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors"
          >
            <Download size={16} />
            Download JSON
          </button>
          {cloudCompareEnabled && (
            <button
              onClick={handleCloudSummarize}
              disabled={cloudLoading}
              className="flex items-center gap-2 px-5 py-2.5 border border-blue-300 text-blue-700 rounded-lg text-sm font-medium hover:bg-blue-50 transition-colors disabled:opacity-50"
            >
              {cloudLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Cloud size={16} />
              )}
              {cloudLoading ? "Summarizing..." : "Compare with Claude"}
            </button>
          )}
        </div>
      )}

      {/* Claude cloud comparison */}
      {cloudSummary && (
        <div className="mb-6 bg-blue-50 border border-blue-200 rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-heading text-lg text-blue-900 flex items-center gap-2">
              <Cloud size={18} className="text-blue-600" />
              Claude Summary (online comparison)
            </h2>
            {cloudMeta && (
              <span className="text-xs text-blue-400">
                {cloudMeta.model} &middot; {cloudMeta.tokens_in}+
                {cloudMeta.tokens_out} tokens
              </span>
            )}
          </div>
          <div className="text-sm text-blue-900 leading-relaxed whitespace-pre-line">
            {cloudSummary}
          </div>
        </div>
      )}

      {/* Stats row — common to all types */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatsCard label="Speakers" value={String(stats.speakers)} />
        <StatsCard label="Segments" value={String(stats.segments)} />
        <StatsCard
          label="Duration"
          value={stats.duration > 0 ? formatTime(stats.duration) : "—"}
        />
        <StatsCard
          label="Status"
          value={job?.state === "completed" ? "OK" : (job?.state ?? "—")}
        />
      </div>

      {/* Body — dispatched per meeting type */}
      {protocol && isWrappedProtocol(protocol) ? (
        protocol.meeting_type === "administrative" ? (
          <AdminProtocolView payload={protocol.payload} />
        ) : protocol.meeting_type === "court_hearing" ? (
          <CourtProtocolView payload={protocol.payload} />
        ) : protocol.meeting_type === "client_meeting" ? (
          <ClientProtocolView payload={protocol.payload} />
        ) : (
          <InterviewProtocolView payload={protocol.payload} />
        )
      ) : protocol ? (
        <LegacyProtocolView protocol={protocol as MeetingProtocol} />
      ) : (
        !error && (
          <div className="text-center py-16 text-epam-gray-400 text-sm">
            Protocol data not available yet. The meeting may still be
            processing.
          </div>
        )
      )}

      {/* Back button */}
      <div className="mt-8">
        <button
          onClick={() => navigate("/meetings")}
          className="flex items-center gap-2 text-sm text-epam-gray-500 hover:text-epam-gray-700 transition-colors"
        >
          <ChevronLeft size={16} />
          Back to meetings
        </button>
      </div>
    </div>
  );
}

// =====================================================================
// Stats card
// =====================================================================

function StatsCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center p-3 bg-white border border-epam-gray-200 rounded-lg">
      <p className="text-2xl font-bold text-epam-gray-900">{value}</p>
      <p className="text-xs text-epam-gray-500 mt-1">{label}</p>
    </div>
  );
}

// =====================================================================
// Stenogram (shared between Admin and Court)
// =====================================================================

function StenogramView({ turns }: { turns: StenogramTurn[] }) {
  if (!turns || turns.length === 0) return null;
  // Group consecutive same-speaker turns for readability.
  const groups: { speaker: string; lines: string[]; start: number }[] = [];
  for (const t of turns) {
    const last = groups[groups.length - 1];
    if (last && last.speaker === t.speaker) {
      last.lines.push(t.text);
    } else {
      groups.push({
        speaker: t.speaker,
        lines: [t.text],
        start: t.start_s ?? 0,
      });
    }
  }
  return (
    <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
      <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
        <MessageSquare size={18} className="text-epam-red" />
        Стенограмма
      </h2>
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {groups.map((g, idx) => (
          <div key={idx}>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium text-epam-red">
                {g.speaker}
              </span>
              {g.start > 0 && (
                <span className="text-xs text-epam-gray-400">
                  {formatTime(g.start)}
                </span>
              )}
            </div>
            <p className="text-sm text-epam-gray-700 leading-relaxed pl-2 border-l-2 border-epam-gray-200">
              {g.lines.join(" ")}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

// =====================================================================
// Reusable item-list cards (admin flat schema)
// =====================================================================

function FlatItemList({
  title,
  items,
  icon,
  iconColor,
  showKind = false,
  showOwner = false,
  showDeadline = false,
  showTimestamp = false,
}: {
  title: string;
  items: FlatProtocolItem[];
  icon: React.ReactNode;
  iconColor: string;
  showKind?: boolean;
  showOwner?: boolean;
  showDeadline?: boolean;
  showTimestamp?: boolean;
}) {
  if (!items || items.length === 0) return null;
  return (
    <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
      <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
        <span className={iconColor}>{icon}</span>
        {title}
      </h2>
      <div className="space-y-3">
        {items.map((it, i) => {
          const isLow = it.confidence === "low";
          const kindLabels: Record<string, string> = {
            decision: "Решение",
            task: "Поручение",
            open_question: "Открытый вопрос",
            risk: "Риск",
            thesis: "Тезис",
          };
          // Low-confidence items get a left border + a "?" badge so
          // the user knows to verify before forwarding the protocol.
          // Sprint 2026-05-04 — replaces the previous behaviour of
          // silently dropping evidence-light items in the verifier.
          return (
            <div
              key={i}
              className={`text-sm border-l-2 pl-3 ${
                isLow ? "border-amber-400 bg-amber-50/30" : "border-epam-gray-200"
              }`}
            >
              <div className="flex items-start gap-2">
                {isLow && (
                  <span
                    className="shrink-0 mt-0.5 w-5 h-5 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-xs font-bold"
                    title="Низкая уверенность — проверьте перед отправкой"
                  >
                    ?
                  </span>
                )}
                <p className="text-epam-gray-800 flex-1">
                  {showKind && it.kind && (
                    <span className="font-semibold text-epam-red mr-1">
                      [{kindLabels[it.kind] || "Тезис"}]
                    </span>
                  )}
                  {it.text}
                </p>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1 text-xs text-epam-gray-500 ml-7">
                {it.speaker && (
                  <span>
                    Кто: <strong>{it.speaker}</strong>
                  </span>
                )}
                {showOwner && it.owner && (
                  <span>
                    Исполнитель: <strong>{it.owner}</strong>
                  </span>
                )}
                {showDeadline && it.deadline && (
                  <span>
                    Срок: <strong>{it.deadline}</strong>
                  </span>
                )}
                {showTimestamp && it.timestamp && (
                  <span>
                    Таймкод: <strong>{it.timestamp}</strong>
                  </span>
                )}
                {it.evidence && (
                  <span className="italic text-epam-gray-400">
                    «{it.evidence}»
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// =====================================================================
// Administrative — flat 9-section format (T3.3)
// =====================================================================

function AdminProtocolView({ payload }: { payload: AdministrativePayload }) {
  const useTopicSegmented = payload.topic_summaries && payload.topic_summaries.length > 0;
  const usePractical = payload.items && payload.items.length > 0;

  return (
    <>
      {/* Goal + Summary */}
      {!usePractical && (payload.meeting_goal || payload.summary) && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <FileText size={18} className="text-epam-red" />
            Краткое содержание
          </h2>
          {payload.meeting_goal && (
            <div className="mb-3">
              <p className="text-xs text-epam-gray-500 mb-1">Цель встречи</p>
              <p className="text-sm font-medium text-epam-gray-900">
                {payload.meeting_goal}
              </p>
            </div>
          )}
          {payload.summary && (
            <p className="text-sm text-epam-gray-700 leading-relaxed whitespace-pre-line">
              {payload.summary}
            </p>
          )}
        </div>
      )}

      {/* Participants — admin payload has them as plain strings */}
      {payload.participants && payload.participants.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
            <Users size={18} className="text-epam-red" />
            Участники
          </h2>
          <div className="flex flex-wrap gap-2">
            {payload.participants.map((name, i) => (
              <span
                key={i}
                className="px-3 py-1 bg-epam-gray-100 text-epam-gray-800 rounded-full text-sm"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Topic-segmented (Sprint 2026-04-30) — each topic is a self-
       * contained mini-protocol with its own discussion + decisions
       * + tasks + open_questions. Risks remain at the bottom because
       * they typically cross-cut topics. */}
      {usePractical ? (
        <FlatItemList
          title="Принятые решения и рабочие тезисы"
          items={payload.items || []}
          icon={<CheckCircle size={18} />}
          iconColor="text-emerald-600"
          showKind
          showOwner
          showDeadline
          showTimestamp
        />
      ) : useTopicSegmented ? (
        <>
          {payload.topic_summaries.map((topic, idx) => (
            <div
              key={idx}
              className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6"
            >
              <h2 className="font-heading text-lg text-epam-red mb-3">
                {idx + 1}. {topic.name || "Тема"}
              </h2>
              {topic.discussion && (
                <p className="text-sm text-epam-gray-700 leading-relaxed whitespace-pre-line mb-4">
                  {topic.discussion}
                </p>
              )}
              {topic.decisions && topic.decisions.length > 0 && (
                <FlatItemList
                  title="Решения"
                  items={topic.decisions}
                  icon={<CheckCircle size={16} />}
                  iconColor="text-emerald-600"
                />
              )}
              {topic.tasks && topic.tasks.length > 0 && (
                <FlatItemList
                  title="Задачи и поручения"
                  items={topic.tasks}
                  icon={<Clock size={16} />}
                  iconColor="text-blue-600"
                  showOwner
                  showDeadline
                />
              )}
              {topic.open_questions && topic.open_questions.length > 0 && (
                <FlatItemList
                  title="Открытые вопросы"
                  items={topic.open_questions}
                  icon={<HelpCircle size={16} />}
                  iconColor="text-amber-600"
                />
              )}
              {!topic.discussion &&
                (!topic.decisions || topic.decisions.length === 0) &&
                (!topic.tasks || topic.tasks.length === 0) &&
                (!topic.open_questions ||
                  topic.open_questions.length === 0) && (
                  <p className="text-xs text-epam-gray-500 italic">
                    По данной теме нет извлечённых решений, задач или
                    вопросов.
                  </p>
                )}
            </div>
          ))}
          {/* Risks — cross-cutting */}
          <FlatItemList
            title="Риски и блокеры"
            items={payload.risks}
            icon={<AlertTriangle size={18} />}
            iconColor="text-red-600"
          />
        </>
      ) : (
        <>
          {/* Flat fallback (T3.3 layout, when topic-segmented is empty) */}
          {payload.topics && payload.topics.length > 0 && (
            <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
              <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
                Основные темы
              </h2>
              <ul className="space-y-2">
                {payload.topics.map((t, i) => (
                  <li
                    key={i}
                    className="flex gap-2 text-sm text-epam-gray-700"
                  >
                    <span className="text-epam-red shrink-0 mt-0.5">
                      &#8226;
                    </span>
                    <span>{t}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <FlatItemList
            title="Принятые решения"
            items={payload.decisions}
            icon={<CheckCircle size={18} />}
            iconColor="text-emerald-600"
          />
          <FlatItemList
            title="Задачи и поручения"
            items={payload.tasks}
            icon={<Clock size={18} />}
            iconColor="text-blue-600"
            showOwner
            showDeadline
          />
          <FlatItemList
            title="Открытые вопросы"
            items={payload.open_questions}
            icon={<HelpCircle size={18} />}
            iconColor="text-amber-600"
          />
          <FlatItemList
            title="Риски и блокеры"
            items={payload.risks}
            icon={<AlertTriangle size={18} />}
            iconColor="text-red-600"
          />
        </>
      )}

      {!usePractical && <StenogramView turns={payload.turns} />}
    </>
  );
}

// =====================================================================
// Court hearing — stenogram + summary header
// =====================================================================

function CourtProtocolView({ payload }: { payload: CourtHearingPayload }) {
  return (
    <>
      {/* Case header */}
      {(payload.case_number || payload.hearing_date || payload.summary) && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <FileText size={18} className="text-epam-red" />
            Краткое содержание
          </h2>
          {(payload.case_number || payload.hearing_date) && (
            <div className="flex flex-wrap gap-x-6 gap-y-1 mb-3 text-xs text-epam-gray-500">
              {payload.case_number && (
                <span>
                  Дело: <strong>{payload.case_number}</strong>
                </span>
              )}
              {payload.hearing_date && (
                <span>
                  Дата: <strong>{payload.hearing_date}</strong>
                </span>
              )}
            </div>
          )}
          {payload.summary && (
            <p className="text-sm text-epam-gray-700 leading-relaxed whitespace-pre-line">
              {payload.summary}
            </p>
          )}
        </div>
      )}

      {/* Participants */}
      {payload.participants && payload.participants.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
            <Users size={18} className="text-epam-red" />
            Участники
          </h2>
          <div className="flex flex-wrap gap-2">
            {payload.participants.map((name, i) => (
              <span
                key={i}
                className="px-3 py-1 bg-epam-gray-100 text-epam-gray-800 rounded-full text-sm"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      <StenogramView turns={payload.turns} />
    </>
  );
}

// =====================================================================
// Client meeting
// =====================================================================

function ClientProtocolView({ payload }: { payload: ClientMeetingPayload }) {
  return (
    <>
      {/* Summary */}
      {(payload.client_name || payload.summary) && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <FileText size={18} className="text-epam-red" />
            Краткое содержание
          </h2>
          {payload.client_name && (
            <p className="text-xs text-epam-gray-500 mb-2">
              Клиент: <strong>{payload.client_name}</strong>
            </p>
          )}
          {payload.summary && (
            <p className="text-sm text-epam-gray-700 leading-relaxed whitespace-pre-line">
              {payload.summary}
            </p>
          )}
        </div>
      )}

      {/* EPAM + client representatives */}
      {((payload.epam_representatives &&
        payload.epam_representatives.length > 0) ||
        (payload.client_representatives &&
          payload.client_representatives.length > 0)) && (
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-white border border-epam-gray-200 rounded-xl p-6">
            <h3 className="font-heading text-sm text-epam-gray-900 mb-3">
              Представители EPAM
            </h3>
            <div className="flex flex-wrap gap-2">
              {(payload.epam_representatives || []).map((n, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-epam-gray-100 text-epam-gray-800 rounded-full text-sm"
                >
                  {n}
                </span>
              ))}
            </div>
          </div>
          <div className="bg-white border border-epam-gray-200 rounded-xl p-6">
            <h3 className="font-heading text-sm text-epam-gray-900 mb-3">
              Представители клиента
            </h3>
            <div className="flex flex-wrap gap-2">
              {(payload.client_representatives || []).map((n, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-epam-gray-100 text-epam-gray-800 rounded-full text-sm"
                >
                  {n}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Topics */}
      {payload.topics && payload.topics.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
            Темы обсуждения
          </h2>
          <div className="space-y-3">
            {payload.topics.map((t, i) => (
              <div
                key={i}
                className="text-sm border-l-2 border-epam-gray-200 pl-3"
              >
                <p className="text-epam-gray-800">{t.text}</p>
                {t.speaker && (
                  <p className="text-xs text-epam-gray-500 mt-1">
                    {t.speaker}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Client requests */}
      {payload.client_requests && payload.client_requests.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <HelpCircle size={18} className="text-amber-600" />
            Запросы клиента
          </h2>
          <div className="space-y-3">
            {payload.client_requests.map((r, i) => (
              <div
                key={i}
                className="text-sm border-l-2 border-amber-200 pl-3"
              >
                <p className="text-epam-gray-800">{r.text}</p>
                {r.speaker && (
                  <p className="text-xs text-epam-gray-500 mt-1">
                    {r.speaker}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Commitments */}
      {payload.commitments && payload.commitments.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <CheckCircle size={18} className="text-emerald-600" />
            Обязательства
          </h2>
          <div className="space-y-3">
            {payload.commitments.map((c, i) => (
              <div
                key={i}
                className={`text-sm border-l-2 pl-3 ${
                  c.party === "epam"
                    ? "border-epam-red"
                    : "border-blue-300"
                }`}
              >
                <div className="flex items-start gap-2">
                  <span
                    className={`text-xs font-medium uppercase shrink-0 mt-0.5 ${
                      c.party === "epam"
                        ? "text-epam-red"
                        : "text-blue-600"
                    }`}
                  >
                    {c.party === "epam" ? "EPAM" : "Client"}
                  </span>
                  <p className="text-epam-gray-800 flex-1">{c.text}</p>
                </div>
                <div className="flex flex-wrap gap-x-4 mt-1 text-xs text-epam-gray-500">
                  {c.speaker && <span>{c.speaker}</span>}
                  {c.deadline && (
                    <span>
                      Срок: <strong>{c.deadline}</strong>
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Follow-ups */}
      {payload.follow_ups && payload.follow_ups.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <Clock size={18} className="text-blue-600" />
            Последующие действия
          </h2>
          <div className="space-y-3">
            {payload.follow_ups.map((f, i) => (
              <div
                key={i}
                className="text-sm border-l-2 border-blue-200 pl-3"
              >
                <p className="text-epam-gray-800">{f.text}</p>
                <div className="flex flex-wrap gap-x-4 mt-1 text-xs text-epam-gray-500">
                  {f.speaker && <span>{f.speaker}</span>}
                  {f.deadline && (
                    <span>
                      Срок: <strong>{f.deadline}</strong>
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Parked questions */}
      {payload.parked_questions && payload.parked_questions.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
            Отложенные вопросы
          </h2>
          <ul className="space-y-2">
            {payload.parked_questions.map((q, i) => (
              <li
                key={i}
                className="flex gap-2 text-sm text-epam-gray-700"
              >
                <span className="text-amber-500 shrink-0 mt-0.5">?</span>
                <span>{q}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

// =====================================================================
// Interview — internal + external reports
// =====================================================================

function InterviewProtocolView({ payload }: { payload: InterviewPayload }) {
  const internal = payload.internal;
  const external = payload.external;
  return (
    <>
      {/* INTERNAL report (confidential, scorecard) */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 mb-6">
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle size={18} className="text-amber-700" />
          <h2 className="font-heading text-lg text-amber-900">
            Внутренний отчёт (конфиденциально)
          </h2>
        </div>
        {internal.role && (
          <p className="text-xs text-amber-800 mb-1">
            Должность: <strong>{internal.role}</strong>
          </p>
        )}
        {internal.candidate_name && (
          <p className="text-xs text-amber-800 mb-3">
            Кандидат: <strong>{internal.candidate_name}</strong>
          </p>
        )}
        {internal.summary && (
          <p className="text-sm text-amber-900 leading-relaxed whitespace-pre-line mb-4">
            {internal.summary}
          </p>
        )}

        <ObservationList
          title="Сильные стороны"
          items={internal.strengths}
          color="text-emerald-700"
        />
        <ObservationList
          title="Опасения"
          items={internal.concerns}
          color="text-red-700"
        />
        <ObservationList
          title="Технические сигналы"
          items={internal.technical_signals}
          color="text-blue-700"
        />
        <ObservationList
          title="Коммуникация"
          items={internal.communication_signals}
          color="text-indigo-700"
        />

        {internal.open_questions_for_next_round &&
          internal.open_questions_for_next_round.length > 0 && (
            <div className="mt-4">
              <h3 className="font-heading text-sm text-amber-900 mb-2">
                Вопросы для следующего раунда
              </h3>
              <ul className="space-y-1">
                {internal.open_questions_for_next_round.map((q, i) => (
                  <li
                    key={i}
                    className="text-sm text-amber-900 flex gap-2"
                  >
                    <span className="text-amber-600 shrink-0">?</span>
                    <span>{q}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
      </div>

      {/* EXTERNAL report (developmental) */}
      <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
        <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
          <FileText size={18} className="text-epam-red" />
          Отчёт для кандидата
        </h2>
        {external.candidate_name && (
          <p className="text-xs text-epam-gray-500 mb-3">
            Кандидат: <strong>{external.candidate_name}</strong>
          </p>
        )}
        {external.speaking_style_notes &&
          external.speaking_style_notes.length > 0 && (
            <div className="mb-4">
              <h3 className="font-heading text-sm text-epam-gray-900 mb-2">
                Особенности речи
              </h3>
              <ul className="space-y-1">
                {external.speaking_style_notes.map((n, i) => (
                  <li
                    key={i}
                    className="text-sm text-epam-gray-700 flex gap-2"
                  >
                    <span className="text-epam-red shrink-0 mt-0.5">
                      &#8226;
                    </span>
                    <span>{n}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        <ObservationList
          title="Сильнейшие моменты"
          items={external.strongest_moments}
          color="text-emerald-700"
        />
        {external.areas_to_develop &&
          external.areas_to_develop.length > 0 && (
            <div className="mb-4">
              <h3 className="font-heading text-sm text-epam-gray-900 mb-2">
                Области для развития
              </h3>
              <ul className="space-y-1">
                {external.areas_to_develop.map((a, i) => (
                  <li
                    key={i}
                    className="text-sm text-epam-gray-700 flex gap-2"
                  >
                    <span className="text-blue-500 shrink-0 mt-0.5">→</span>
                    <span>{a}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        {external.closing_note && (
          <p className="text-sm text-epam-gray-700 leading-relaxed whitespace-pre-line mt-4 italic">
            {external.closing_note}
          </p>
        )}
      </div>
    </>
  );
}

function ObservationList({
  title,
  items,
  color,
}: {
  title: string;
  items: { observation: string; evidence: string }[];
  color: string;
}) {
  if (!items || items.length === 0) return null;
  return (
    <div className="mb-4">
      <h3 className={`font-heading text-sm mb-2 ${color}`}>{title}</h3>
      <div className="space-y-2">
        {items.map((it, i) => (
          <div key={i} className="text-sm">
            <p className="text-epam-gray-800">{it.observation}</p>
            {it.evidence && (
              <p className="text-xs text-epam-gray-500 mt-0.5 italic pl-3 border-l-2 border-epam-gray-200">
                «{it.evidence}»
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// =====================================================================
// Legacy — pre-T3.3 MeetingProtocol shape
// =====================================================================

function LegacyProtocolView({ protocol }: { protocol: MeetingProtocol }) {
  const transcript = protocol.transcript || [];

  // Group consecutive transcript segments by speaker.
  const groups: { speaker: string; lines: string[]; start: number }[] = [];
  for (const seg of transcript) {
    const label = seg.speaker_name || seg.speaker_id;
    const last = groups[groups.length - 1];
    if (last && last.speaker === label) {
      last.lines.push(seg.text);
    } else {
      groups.push({ speaker: label, lines: [seg.text], start: seg.start });
    }
  }

  return (
    <>
      {/* Summary */}
      {protocol.summary && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3 flex items-center gap-2">
            <FileText size={18} className="text-epam-red" />
            Summary
          </h2>
          {protocol.topic && (
            <p className="text-sm font-medium text-epam-gray-900 mb-3">
              {protocol.topic}
            </p>
          )}
          <p className="text-sm text-epam-gray-700 leading-relaxed whitespace-pre-line">
            {protocol.summary}
          </p>
        </div>
      )}

      {/* Participants with speaking stats */}
      {protocol.participants && protocol.participants.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
            <Users size={18} className="text-epam-red" />
            Participants
          </h2>
          <div className="space-y-3">
            {protocol.participants.map((p) => (
              <div key={p.speaker_id} className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-epam-red/10 text-epam-red flex items-center justify-center text-xs font-bold shrink-0">
                  {(p.speaker_name || p.speaker_id).charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-epam-gray-900 truncate">
                    {p.speaker_name || p.speaker_id}
                  </p>
                  <p className="text-xs text-epam-gray-400">
                    {formatDuration(p.speaking_time)} speaking
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <div className="w-24 h-2 bg-epam-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-epam-red rounded-full"
                      style={{
                        width: `${Math.min(p.speaking_share, 100)}%`,
                      }}
                    />
                  </div>
                  <span className="text-xs text-epam-gray-500 w-10 text-right">
                    {p.speaking_share.toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key topics */}
      {protocol.key_topics && protocol.key_topics.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
            Key Topics
          </h2>
          <ul className="space-y-2">
            {protocol.key_topics.map((t, i) => (
              <li
                key={i}
                className="flex gap-2 text-sm text-epam-gray-700"
              >
                <span className="text-epam-red shrink-0 mt-0.5">&#8226;</span>
                <span>{t.content || t.title}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Tasks */}
      {protocol.tasks && protocol.tasks.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
            Tasks
          </h2>
          <div className="space-y-3">
            {protocol.tasks.map((t, i) => (
              <div
                key={i}
                className="flex gap-3 text-sm border-l-2 border-epam-red pl-3"
              >
                <div className="flex-1">
                  <p className="text-epam-gray-800">{t.text}</p>
                  <div className="flex gap-4 mt-1 text-xs text-epam-gray-500">
                    {t.assignee && (
                      <span>
                        Assignee: <strong>{t.assignee}</strong>
                      </span>
                    )}
                    {t.deadline && (
                      <span>
                        Deadline: <strong>{t.deadline}</strong>
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Open questions */}
      {protocol.open_questions && protocol.open_questions.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
            Open Questions
          </h2>
          <ul className="space-y-2">
            {protocol.open_questions.map((q, i) => (
              <li
                key={i}
                className="flex gap-2 text-sm text-epam-gray-700"
              >
                <span className="text-amber-500 shrink-0 mt-0.5">?</span>
                <span>{q}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Decisions */}
      {protocol.decisions && protocol.decisions.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-3">
            Decisions
          </h2>
          <ul className="space-y-2">
            {protocol.decisions.map((d, i) => (
              <li
                key={i}
                className="flex gap-2 text-sm text-epam-gray-700"
              >
                <span className="text-green-600 shrink-0 mt-0.5">&#10003;</span>
                <span>{d.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Transcript */}
      {groups.length > 0 && (
        <div className="bg-white border border-epam-gray-200 rounded-xl p-6">
          <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
            <MessageSquare size={18} className="text-epam-red" />
            Transcript
          </h2>
          <div className="space-y-4 max-h-[600px] overflow-y-auto">
            {groups.map((group, idx) => (
              <div key={idx}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-epam-red">
                    {group.speaker}
                  </span>
                  <span className="text-xs text-epam-gray-400">
                    {formatTime(group.start)}
                  </span>
                </div>
                <p className="text-sm text-epam-gray-700 leading-relaxed pl-2 border-l-2 border-epam-gray-200">
                  {group.lines.join(" ")}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
