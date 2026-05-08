/**
 * Meetings list page showing all processed and in-progress meetings.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FileText,
  Loader2,
  AlertCircle,
  Trash2,
  Eye,
  Clock,
  CheckCircle,
  XCircle,
  RefreshCw,
  Upload,
} from "lucide-react";
import { listMeetings, deleteMeeting } from "../api/client";
import type { MeetingJob, PipelineState } from "../types/api";

function stateLabel(state: PipelineState): {
  text: string;
  color: string;
  icon: typeof CheckCircle;
} {
  switch (state) {
    case "completed":
      return { text: "Completed", color: "text-green-600 bg-green-50", icon: CheckCircle };
    case "error":
      return { text: "Error", color: "text-red-600 bg-red-50", icon: XCircle };
    case "uploaded":
      return { text: "Uploaded", color: "text-blue-600 bg-blue-50", icon: Clock };
    default:
      return { text: "Processing", color: "text-amber-600 bg-amber-50", icon: RefreshCw };
  }
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function MeetingsPage() {
  const navigate = useNavigate();
  const [meetings, setMeetings] = useState<MeetingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  async function load() {
    try {
      const data = await listMeetings();
      setMeetings(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load meetings");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(jobId: string) {
    if (!confirm("Delete this meeting and all associated data?")) return;
    setDeleting(jobId);
    try {
      await deleteMeeting(jobId);
      setMeetings((prev) => prev.filter((m) => m.id !== jobId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  }

  function handleView(job: MeetingJob) {
    if (job.state === "completed") {
      navigate(`/protocol/${job.id}`);
    } else if (job.state === "error") {
      navigate(`/processing/${job.id}`);
    } else {
      navigate(`/processing/${job.id}`);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={24} className="animate-spin text-epam-red" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-heading text-2xl text-epam-gray-900">
            Meetings
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            {meetings.length} meeting{meetings.length !== 1 ? "s" : ""}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={load}
            className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
            title="Refresh"
          >
            <RefreshCw size={18} />
          </button>
          <button
            onClick={() => navigate("/upload")}
            className="flex items-center gap-2 px-4 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors"
          >
            <Upload size={16} />
            New Upload
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {meetings.length === 0 ? (
        <div className="text-center py-20">
          <FileText size={48} className="mx-auto text-epam-gray-300 mb-4" />
          <p className="text-epam-gray-500 mb-4">No meetings yet</p>
          <button
            onClick={() => navigate("/upload")}
            className="px-5 py-2.5 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors"
          >
            Upload your first recording
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {meetings.map((job) => {
            const status = stateLabel(job.state);
            const StatusIcon = status.icon;

            return (
              <div
                key={job.id}
                className="flex items-center gap-4 p-4 bg-white border border-epam-gray-200 rounded-xl hover:border-epam-gray-300 transition-colors"
              >
                <FileText size={20} className="text-epam-gray-400 shrink-0" />

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-epam-gray-900 truncate">
                    {job.filename}
                  </p>
                  <p className="text-xs text-epam-gray-400 mt-0.5">
                    {formatDate(job.created_at)} &middot; ID: {job.id.slice(0, 8)}
                  </p>
                </div>

                {/* Progress for active jobs */}
                {job.state !== "completed" && job.state !== "error" && job.state !== "uploaded" && (
                  <div className="w-20">
                    <div className="h-1.5 bg-epam-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-epam-red rounded-full transition-all"
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-epam-gray-400 mt-1 text-center">
                      {Math.round(job.progress)}%
                    </p>
                  </div>
                )}

                {/* Status badge */}
                <span
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${status.color}`}
                >
                  <StatusIcon size={12} />
                  {status.text}
                </span>

                {/* Actions */}
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => handleView(job)}
                    className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
                    title="View"
                  >
                    <Eye size={16} />
                  </button>
                  <button
                    onClick={() => handleDelete(job.id)}
                    disabled={deleting === job.id}
                    className="p-2 text-epam-gray-400 hover:text-red-500 transition-colors disabled:opacity-50"
                    title="Delete"
                  >
                    {deleting === job.id ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <Trash2 size={16} />
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
