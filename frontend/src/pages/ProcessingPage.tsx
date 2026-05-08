/**
 * Real-time processing progress page with stepper visualization.
 *
 * Connects via WebSocket to receive live updates from the pipeline.
 * Shows each processing stage with progress indicators.
 */

import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  CheckCircle,
  Circle,
  Loader2,
  AlertTriangle,
  Clock,
  ArrowRight,
  RotateCw,
  FileText,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useWebSocket } from "../hooks/useWebSocket";
import {
  getMeetingStatus,
  getTranscript,
  retryMeeting,
} from "../api/client";
import GpuStatus from "../components/GpuStatus";
import { usePreferences } from "../context/PreferencesContext";
import type { PipelineState, MeetingJob, AlignedSegment } from "../types/api";

/** Ordered pipeline stages for the stepper. */
const STAGES: { key: PipelineState; label: string }[] = [
  { key: "uploaded", label: "Uploaded" },
  { key: "preprocessing", label: "Preprocessing" },
  { key: "transcribing", label: "Transcription (ASR)" },
  { key: "diarizing", label: "Speaker Diarization" },
  { key: "aligning", label: "Alignment" },
  { key: "summarizing", label: "Summarization (LLM)" },
  { key: "qa_validating", label: "Quality Check" },
  { key: "formatting", label: "Protocol Generation" },
  { key: "completed", label: "Completed" },
];

function stageIndex(state: PipelineState): number {
  return STAGES.findIndex((s) => s.key === state);
}

function formatEta(seconds: number | null): string {
  if (seconds === null || seconds <= 0) return "";
  if (seconds < 60) return `~${Math.ceil(seconds)}s remaining`;
  const min = Math.floor(seconds / 60);
  const sec = Math.ceil(seconds % 60);
  return `~${min}m ${sec}s remaining`;
}

export default function ProcessingPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { resolvedTheme } = usePreferences();
  const isDark = resolvedTheme === "dark";
  const ws = useWebSocket(jobId ?? null);

  const [job, setJob] = useState<MeetingJob | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [previewSegments, setPreviewSegments] = useState<AlignedSegment[]>([]);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewAttempted, setPreviewAttempted] = useState(false);
  const [startTime] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);

  // Fallback: poll status every 3s if WebSocket is not connected
  useEffect(() => {
    if (!jobId) return;

    let timer: ReturnType<typeof setInterval> | null = null;

    async function poll() {
      try {
        const data = await getMeetingStatus(jobId!);
        setJob(data);
        setPollError(null);
      } catch (err) {
        setPollError(err instanceof Error ? err.message : "Failed to fetch status");
      }
    }

    poll(); // initial fetch

    // Always poll as safety net — WebSocket may not push updates
    timer = setInterval(poll, 8000);

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [jobId, ws.connected]);

  function formatElapsed(secs: number): string {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  // Use WebSocket data when available, fall back to polled job data
  const currentState: PipelineState = ws.connected
    ? ws.state
    : job?.state ?? "uploaded";
  const currentProgress = ws.connected ? ws.progress : job?.progress ?? 0;
  const currentStage = ws.connected
    ? ws.currentStage
    : job?.current_stage ?? "";
  const etaSeconds = ws.connected ? ws.etaSeconds : job?.eta_seconds ?? null;
  const isError = currentState === "error";
  const isCompleted = currentState === "completed";

  function stepIconClass(isFailed: boolean, isDone: boolean, isActive: boolean) {
    if (isFailed) {
      return isDark
        ? "bg-red-950/70 text-red-300 ring-1 ring-red-800"
        : "bg-red-100 text-red-600";
    }
    if (isDone) {
      return isDark
        ? "bg-emerald-950/70 text-emerald-300 ring-1 ring-emerald-800"
        : "bg-green-100 text-green-600";
    }
    if (isActive) {
      return isDark
        ? "bg-epam-red-light text-epam-red ring-1 ring-epam-red/40"
        : "bg-epam-red/10 text-epam-red";
    }
    return isDark
      ? "bg-epam-gray-100 text-epam-gray-400 ring-1 ring-epam-gray-300"
      : "bg-epam-gray-100 text-epam-gray-400";
  }

  function stepLabelClass(isFailed: boolean, isDone: boolean, isActive: boolean) {
    if (isFailed) return isDark ? "text-red-300" : "text-red-600";
    if (isDone) return isDark ? "text-emerald-300" : "text-green-700";
    if (isActive) return "text-epam-gray-900";
    return "text-epam-gray-400";
  }

  // Elapsed time counter — wall-clock reference for accuracy
  useEffect(() => {
    if (isCompleted || isError) return;
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 500);
    return () => clearInterval(timer);
  }, [isCompleted, isError, startTime]);

  const activeIdx = stageIndex(currentState);

  // Try to fetch transcript preview once past alignment stage
  const pastAlignment = stageIndex(currentState) > stageIndex("aligning");
  useEffect(() => {
    if (!jobId || !pastAlignment || previewAttempted) return;
    setPreviewAttempted(true);
    setPreviewLoading(true);
    getTranscript(jobId)
      .then((segs) => {
        setPreviewSegments(segs);
        setPreviewOpen(true);
      })
      .catch(() => {
        // Backend may not serve transcript mid-pipeline yet
      })
      .finally(() => setPreviewLoading(false));
  }, [jobId, pastAlignment, previewAttempted]);

  return (
    <div className="max-w-2xl mx-auto px-6 py-10">
      <h1 className="font-heading text-2xl text-epam-gray-900 mb-2">
        Processing Meeting
      </h1>
      <p className="text-sm text-epam-gray-500 mb-8">
        {job?.filename ?? "Audio file"} &middot;{" "}
        {!isCompleted && !isError && (
          <span className="text-epam-gray-600 font-mono">{formatElapsed(elapsed)} elapsed</span>
        )}
        {isCompleted && (
          <span className="text-emerald-500 font-mono">{formatElapsed(elapsed)} total</span>
        )}
        {isError && (
          <span className="text-red-600 font-mono">failed at {formatElapsed(elapsed)}</span>
        )}
        {" "}&middot;{" "}
        {ws.connected ? (
          <span className="text-emerald-500">Live</span>
        ) : (
          <span className="text-amber-600">Polling</span>
        )}
      </p>

      {/* Overall progress bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-epam-gray-700">
            {isCompleted
              ? "Processing complete"
              : isError
                ? "Processing failed"
                : `${currentProgress.toFixed(1)}% complete`}
          </span>
          {etaSeconds !== null && !isCompleted && !isError && (
            <span className="flex items-center gap-1 text-xs text-epam-gray-400">
              <Clock size={12} />
              {formatEta(etaSeconds)}
            </span>
          )}
        </div>
        <div className="h-2 bg-epam-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isError
                ? "bg-red-500"
                : isCompleted
                  ? "bg-emerald-500"
                  : "bg-epam-red"
            }`}
            style={{ width: `${Math.min(currentProgress, 100)}%` }}
          />
        </div>
      </div>

      {/* Pipeline stepper */}
      <div className="space-y-0">
        {STAGES.map((stage, idx) => {
          const isActive = idx === activeIdx;
          const isDone = activeIdx > idx || isCompleted;
          const isFailed = isError && isActive;

          return (
            <div key={stage.key} className="flex items-stretch">
              {/* Icon column */}
              <div className="flex flex-col items-center w-8 shrink-0">
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${stepIconClass(isFailed, isDone, isActive)}`}
                >
                  {isFailed ? (
                    <AlertTriangle size={14} />
                  ) : isDone ? (
                    <CheckCircle size={14} />
                  ) : isActive ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Circle size={14} />
                  )}
                </div>
                {/* Connector line */}
                {idx < STAGES.length - 1 && (
                  <div
                    className={`w-0.5 flex-1 my-1 ${
                      isDone
                        ? isDark
                          ? "bg-emerald-800"
                          : "bg-green-300"
                        : "bg-epam-gray-200"
                    }`}
                  />
                )}
              </div>

              {/* Label */}
              <div className="ml-3 pb-6">
                <p
                  className={`text-sm font-medium ${stepLabelClass(isFailed, isDone, isActive)}`}
                >
                  {stage.label}
                </p>
                {isActive && currentStage && !isCompleted && (
                  <p className="text-xs text-epam-gray-400 mt-0.5">
                    {currentStage}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Transcript preview (available after alignment) */}
      {previewSegments.length > 0 && (
        <div className="mt-6 border border-epam-gray-200 rounded-lg overflow-hidden">
          <button
            onClick={() => setPreviewOpen(!previewOpen)}
            className="w-full flex items-center justify-between px-4 py-3 bg-epam-gray-50 hover:bg-epam-gray-100 transition-colors"
          >
            <span className="flex items-center gap-2 text-sm font-medium text-epam-gray-700">
              <FileText size={16} />
              Transcript Preview ({previewSegments.length} segments)
            </span>
            {previewOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {previewOpen && (
            <div className="max-h-80 overflow-y-auto px-4 py-3 space-y-2 text-sm">
              {previewSegments.map((seg, i) => (
                <div key={i} className="flex gap-3">
                  <span className="shrink-0 text-xs text-epam-gray-400 font-mono w-20 pt-0.5">
                    {Math.floor(seg.start / 60)}:{String(Math.floor(seg.start % 60)).padStart(2, "0")}
                  </span>
                  <span className="shrink-0 text-xs font-medium text-epam-red w-20 pt-0.5">
                    {seg.speaker_name || seg.speaker_id}
                  </span>
                  <span className="text-epam-gray-700">{seg.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {previewLoading && (
        <p className="mt-4 text-xs text-epam-gray-400 flex items-center gap-2">
          <Loader2 size={12} className="animate-spin" /> Loading transcript preview...
        </p>
      )}

      {/* Error details */}
      {isError && jobId && (
        <ErrorBlock
          jobId={jobId}
          message={job?.error || ws.error || "An unexpected error occurred"}
          onUploadAgain={() => navigate("/upload")}
        />
      )}

      {/* Completed actions */}
      {isCompleted && jobId && (
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            onClick={() => navigate(`/transcript/${jobId}`)}
            className="flex items-center gap-2 px-5 py-2.5 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors"
          >
            Review Transcript
            <ArrowRight size={16} />
          </button>
          <button
            onClick={() => navigate(`/protocol/${jobId}`)}
            className="flex items-center gap-2 px-5 py-2.5 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors"
          >
            View Protocol
          </button>
        </div>
      )}

      {pollError && !ws.connected && (
        <p className="mt-4 text-xs text-amber-600">{pollError}</p>
      )}
    </div>
  );
}


/**
 * Error block on the processing page. When the failure looks like a
 * VRAM exhaustion, render a special panel with: live GPU status,
 * specific guidance ("close LM Studio / SD / etc"), and a "Retry
 * without re-uploading" button. For other errors, just show the
 * generic message + Upload Again link.
 */
function ErrorBlock({
  jobId,
  message,
  onUploadAgain,
}: {
  jobId: string;
  message: string;
  onUploadAgain: () => void;
}) {
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);

  const isVram =
    /vram|видеопам|insufficient/i.test(message) ||
    /HTTP\s*503/i.test(message);

  async function handleRetry() {
    setRetrying(true);
    setRetryError(null);
    try {
      await retryMeeting(jobId);
      // Backend reset state and started a new background task. Stay
      // on this page; status polling + WebSocket will pick up the
      // fresh state automatically (`isError` is computed from
      // `currentState`, which updates when polling sees the new
      // "uploaded" state). The error block hides itself when state
      // leaves "error".
    } catch (err) {
      setRetryError(
        err instanceof Error ? err.message : "Retry failed",
      );
      setRetrying(false);
    }
  }

  if (isVram) {
    return (
      <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-start gap-3">
          <AlertTriangle
            size={18}
            className="text-red-600 mt-0.5 shrink-0"
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-red-700 font-medium">
              Не хватило видеопамяти GPU
            </p>
            <p className="text-sm text-red-600 mt-1 leading-relaxed">
              Пайплайн остановился, потому что во время работы другая
              программа заняла слишком много видеопамяти, и нашему
              распознаванию не хватило места.
            </p>

            <div className="mt-3 mb-3">
              <GpuStatus refreshSec={3} />
            </div>

            <p className="text-xs text-red-700 leading-relaxed mb-3">
              Закройте программы, которые могут использовать GPU:
              LM Studio, Ollama (другая сессия), Stable Diffusion,
              ChatGPT/Claude desktop, игры в фоне, браузер с открытым
              AI-чатом. Когда индикатор выше станет зелёным —
              нажмите «Повторить».
            </p>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleRetry}
                disabled={retrying}
                className="flex items-center gap-2 px-4 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors disabled:opacity-50"
              >
                {retrying ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <RotateCw size={14} />
                )}
                Повторить
              </button>
              <button
                onClick={onUploadAgain}
                className="flex items-center gap-2 px-4 py-2 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors"
              >
                Загрузить заново
              </button>
            </div>

            {retryError && (
              <p className="mt-2 text-xs text-red-700">
                Не удалось перезапустить: {retryError}
              </p>
            )}

            <details className="mt-3 text-xs text-red-600">
              <summary className="cursor-pointer">
                Техническая информация
              </summary>
              <p className="mt-1 font-mono break-all">{message}</p>
            </details>
          </div>
        </div>
      </div>
    );
  }

  // Generic error — same UX as before plus a Retry button next to
  // Upload Again, since most transient errors (Ollama hiccup, network
  // blip on a model fetch) are also worth retrying without re-upload.
  return (
    <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
      <p className="text-sm text-red-700 font-medium mb-1">
        Processing Error
      </p>
      <p className="text-sm text-red-600">{message}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          onClick={handleRetry}
          disabled={retrying}
          className="flex items-center gap-2 text-sm text-red-700 hover:text-red-800 font-medium disabled:opacity-50"
        >
          {retrying ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <RotateCw size={14} />
          )}
          Retry
        </button>
        <span className="text-red-300">|</span>
        <button
          onClick={onUploadAgain}
          className="flex items-center gap-2 text-sm text-red-700 hover:text-red-800 font-medium"
        >
          Upload again
        </button>
      </div>
      {retryError && (
        <p className="mt-2 text-xs text-red-700">
          Retry failed: {retryError}
        </p>
      )}
    </div>
  );
}
