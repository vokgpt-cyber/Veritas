/**
 * HTTP client for VERITAS backend API.
 *
 * All requests include JWT token from localStorage when available.
 * The Vite dev server proxies /api and /ws to the backend at :8000.
 */

import type {
  MeetingJob,
  MeetingProtocol,
  MeetingType,
  AlignedSegment,
  SpeakerProfile,
  SystemHealth,
  GpuStatus,
  SystemStats,
  SystemFeatures,
  AuthResponse,
  CurrentUser,
  UserAccount,
  AsrEngineChoice,
  WrappedProtocol,
  TranscriptReview,
} from "../types/api";

const API_BASE = "/api";

/** Storage key for JWT token. */
const TOKEN_KEY = "veritas_token";

/** Get stored JWT token. */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

/** Store JWT token. */
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

/** Remove JWT token. */
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/** Build headers with optional auth. */
function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

/** Generic fetch wrapper with error handling. */
async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: authHeaders(options.headers as Record<string, string>),
  });

  if (response.status === 401) {
    clearToken();
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg || "").join("; ")
          : `Request failed: ${response.status}`;
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// --- Auth ---

/**
 * Authenticate with username and password.
 * Backend may not have auth yet (Block 2), so we handle gracefully.
 */
export async function login(
  username: string,
  password: string,
): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    body: JSON.stringify({ username, password }),
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    // detail can be a string or an array of validation errors
    const detail = body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg || "").join("; ")
          : "Authentication failed";
    throw new Error(message);
  }

  const data: AuthResponse = await response.json();
  setToken(data.access_token);
  return data;
}

export async function getCurrentUser(): Promise<CurrentUser> {
  return request<CurrentUser>("/auth/me");
}

export async function logoutSession(): Promise<{ status: string }> {
  return request("/auth/logout", { method: "POST" });
}

export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<{ status: string }> {
  return request("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
}

export async function listUsers(): Promise<UserAccount[]> {
  return request<UserAccount[]>("/users/");
}

export async function createUserAccount(payload: {
  username: string;
  password: string;
  role: string;
  must_change_password: boolean;
}): Promise<UserAccount> {
  return request<UserAccount>("/users/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function updateUserAccount(
  username: string,
  payload: Partial<Pick<UserAccount, "role" | "is_active" | "must_change_password">>,
): Promise<UserAccount> {
  return request<UserAccount>(`/users/${encodeURIComponent(username)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function resetUserPassword(
  username: string,
  password: string,
): Promise<UserAccount> {
  return request<UserAccount>(
    `/users/${encodeURIComponent(username)}/reset-password`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    },
  );
}

export async function deleteUserAccount(
  username: string,
): Promise<{ status: string; username: string }> {
  return request(`/users/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });
}

/** Check if backend health endpoint is reachable. */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

// --- Meetings ---

/**
 * Upload audio file for processing with optional context.
 *
 * meetingType drives: (a) summarization prompt selection, (b) DOCX
 * template, and (c) ASR engine routing in the orchestrator. Omit to
 * default to "generic" on the backend.
 *
 * Speaker constraints accept either a single `expectedSpeakers` (legacy)
 * OR a `speakerRange: {min, max}` (new bucket UI, T3.1). Pass at most
 * one. Passing nothing leaves pyannote at its default broad range.
 */
export interface UploadOptions {
  meetingType?: MeetingType;
  expectedSpeakers?: number;
  speakerRange?: { min: number; max: number };
  context?: string;
  contextFiles?: File[];
  courtParticipants?: string;
  courtDictionary?: string;
  // Per-meeting opt-in feature toggles. Undefined = use server config
  // default; true/false = override for this run only.
  enableLlmCorrection?: boolean;
  enablePolishPass?: boolean;
  asrEngineOverride?: AsrEngineChoice;
}

export async function uploadAudio(
  file: File,
  meetingTypeOrOptions?: MeetingType | UploadOptions,
  expectedSpeakersLegacy?: number,
  contextLegacy?: string,
  contextFilesLegacy?: File[],
): Promise<MeetingJob> {
  // Backwards-compat shim: callers used to pass positional args
  // (file, meetingType, expectedSpeakers, context, contextFiles).
  // We now accept either positional OR an options object. MeetingType
  // is a string union, options is a plain object → typeof
  // discriminates cleanly.
  let opts: UploadOptions;
  if (typeof meetingTypeOrOptions === "object" && meetingTypeOrOptions !== null) {
    opts = meetingTypeOrOptions;
  } else {
    opts = {
      meetingType: meetingTypeOrOptions,
      expectedSpeakers: expectedSpeakersLegacy,
      context: contextLegacy,
      contextFiles: contextFilesLegacy,
    };
  }

  const formData = new FormData();
  formData.append("file", file);
  if (opts.meetingType) {
    formData.append("meeting_type", opts.meetingType);
  }
  // Speaker constraints. Prefer min/max range when provided; fall
  // back to single expectedSpeakers for the legacy code path.
  if (opts.speakerRange) {
    formData.append(
      "expected_speakers_min",
      String(opts.speakerRange.min),
    );
    formData.append(
      "expected_speakers_max",
      String(opts.speakerRange.max),
    );
  } else if (opts.expectedSpeakers !== undefined) {
    formData.append("expected_speakers", String(opts.expectedSpeakers));
  }
  if (opts.context) {
    formData.append("context", opts.context);
  }
  if (opts.courtParticipants) {
    formData.append("court_participants", opts.courtParticipants);
  }
  if (opts.courtDictionary) {
    formData.append("court_dictionary", opts.courtDictionary);
  }
  if (opts.contextFiles) {
    for (const cf of opts.contextFiles) {
      formData.append("context_files", cf);
    }
  }
  // Per-meeting opt-in toggles. Only send when set; undefined leaves
  // the field absent and the backend falls through to config defaults.
  if (opts.enableLlmCorrection !== undefined) {
    formData.append("enable_llm_correction", String(opts.enableLlmCorrection));
  }
  if (opts.enablePolishPass !== undefined) {
    formData.append("enable_polish_pass", String(opts.enablePolishPass));
  }
  if (opts.asrEngineOverride && opts.asrEngineOverride !== "auto") {
    formData.append("asr_engine_override", opts.asrEngineOverride);
  }

  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}/meetings/upload`, {
    method: "POST",
    body: formData,
    headers,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Upload failed");
  }

  return response.json();
}

/** List all meetings. */
export async function listMeetings(): Promise<MeetingJob[]> {
  return request<MeetingJob[]>("/meetings/");
}

/** Get meeting status. */
export async function getMeetingStatus(jobId: string): Promise<MeetingJob> {
  return request<MeetingJob>(`/meetings/${jobId}/status`);
}

/** Get transcript segments. */
export async function getTranscript(
  jobId: string,
): Promise<AlignedSegment[]> {
  return request<AlignedSegment[]>(`/meetings/${jobId}/transcript`);
}

/** Get speaker-count, voice-suggestion, and dictionary review data. */
export async function getTranscriptReview(
  jobId: string,
): Promise<TranscriptReview> {
  return request<TranscriptReview>(`/meetings/${jobId}/review`);
}

export async function confirmSpeakerReview(
  jobId: string,
): Promise<{ status: string; job_id: string; speaker_review: unknown }> {
  return request(`/meetings/${jobId}/review/confirm`, { method: "POST" });
}

/** Save edited transcript. */
export async function saveTranscript(
  jobId: string,
  segments: AlignedSegment[],
): Promise<{ status: string; job_id: string; segments_count: number }> {
  return request(`/meetings/${jobId}/transcript`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });
}

/** Get protocol download URL. */
export function getProtocolUrl(jobId: string, format: string = "docx"): string {
  return `${API_BASE}/meetings/${jobId}/protocol?format=${format}`;
}

/** Delete meeting. */
export async function deleteMeeting(
  jobId: string,
): Promise<{ status: string; job_id: string }> {
  return request(`/meetings/${jobId}`, { method: "DELETE" });
}

/**
 * Get protocol data as JSON.
 *
 * Returns a union of two shapes — the response is either:
 * - LEGACY `MeetingProtocol` (flat fields: participants, transcript, ...)
 *   for meeting_type=generic and older protocols on disk.
 * - WRAPPED `{meeting_type, payload}` for admin / court_hearing / client /
 *   interview meetings produced by the new code path (T3.3+).
 *
 * The frontend uses `isWrappedProtocol()` from types/api.ts to discriminate
 * and dispatch to the right view. Backend's GET /protocol/data passes the
 * stored JSON through unchanged, so the shape on the wire matches the
 * shape on disk.
 */
export async function getProtocolData(
  jobId: string,
): Promise<MeetingProtocol | WrappedProtocol> {
  return request<MeetingProtocol | WrappedProtocol>(
    `/meetings/${jobId}/protocol/data`,
  );
}

/** Rename a speaker across all transcript segments. */
export async function renameSpeaker(
  jobId: string,
  oldName: string,
  newName: string,
): Promise<{ status: string; renamed_segments: number }> {
  return request(
    `/meetings/${jobId}/speakers/rename?old_name=${encodeURIComponent(oldName)}&new_name=${encodeURIComponent(newName)}`,
    { method: "PUT" },
  );
}

/** Re-run summarization on edited transcript. */
export async function resummarize(
  jobId: string,
): Promise<{ status: string; job_id: string }> {
  return request(`/meetings/${jobId}/resummarize`, { method: "POST" });
}

/** Summarize with Claude API (online, for comparison). */
export async function summarizeCloud(
  jobId: string,
): Promise<{ status: string; summary: string; model: string; tokens_in: number; tokens_out: number }> {
  return request(`/meetings/${jobId}/summarize-cloud`, { method: "POST" });
}

// --- Speakers ---

/** List all registered speakers. */
export async function listSpeakers(): Promise<SpeakerProfile[]> {
  return request<SpeakerProfile[]>("/speakers/");
}

/** Enrol a new voice profile from an audio sample (Block 7).
 *
 * Sends multipart/form-data with name, optional department/position,
 * and the audio blob. Returns the created profile.
 */
export async function enrolSpeaker(opts: {
  name: string;
  department?: string;
  position?: string;
  isAdminDefault?: boolean;
  audio?: Blob;
  audioFilename?: string;
}): Promise<SpeakerProfile> {
  const fd = new FormData();
  fd.append("name", opts.name);
  if (opts.department) fd.append("department", opts.department);
  if (opts.position) fd.append("position", opts.position);
  fd.append("is_admin_default", String(Boolean(opts.isAdminDefault)));
  if (opts.audio) {
    // Provide a filename so the backend's extension whitelist accepts
    // the upload — Blob objects from MediaRecorder otherwise lack one.
    const filename = opts.audioFilename || "voice_sample.webm";
    fd.append("audio", opts.audio, filename);
  }

  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE}/speakers/`, {
    method: "POST",
    body: fd,
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg || "").join("; ")
          : `Enrollment failed: ${response.status}`;
    throw new Error(message);
  }
  return response.json();
}

/** Add an additional voice sample to an existing profile. */
export async function addSpeakerSample(
  speakerId: string,
  audio: Blob,
  audioFilename = "voice_sample.webm",
): Promise<SpeakerProfile> {
  const fd = new FormData();
  fd.append("audio", audio, audioFilename);

  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(
    `${API_BASE}/speakers/${speakerId}/samples`,
    { method: "POST", body: fd, headers },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Sample upload failed: ${response.status}`);
  }
  return response.json();
}

/** Update profile metadata (name, department, position). */
export async function updateSpeakerProfile(
  speakerId: string,
  fields: {
    name?: string;
    department?: string;
    position?: string;
    is_admin_default?: boolean;
  },
): Promise<SpeakerProfile> {
  const fd = new FormData();
  if (fields.name !== undefined) fd.append("name", fields.name);
  if (fields.department !== undefined)
    fd.append("department", fields.department);
  if (fields.position !== undefined) fd.append("position", fields.position);
  if (fields.is_admin_default !== undefined) {
    fd.append("is_admin_default", String(fields.is_admin_default));
  }

  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE}/speakers/${speakerId}`, {
    method: "PUT",
    body: fd,
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Update failed: ${response.status}`);
  }
  return response.json();
}

/** Delete speaker. */
export async function deleteSpeaker(
  speakerId: string,
): Promise<{ status: string; speaker_id: string; name?: string }> {
  return request(`/speakers/${speakerId}`, { method: "DELETE" });
}

// --- System ---

/** Get system health. */
export async function getSystemHealth(): Promise<SystemHealth> {
  return request<SystemHealth>("/system/health");
}

/** Get GPU status. */
export async function getGpuStatus(): Promise<GpuStatus> {
  return request<GpuStatus>("/system/gpu");
}

/** Get system statistics. */
export async function getSystemStats(): Promise<SystemStats> {
  return request<SystemStats>("/system/stats");
}

/** Re-run a failed job. The audio + context stay associated; only
 * the pipeline state is reset and reprocessing is triggered. Backend
 * checks VRAM availability and refuses if another job is active.
 */
export async function retryMeeting(jobId: string): Promise<MeetingJob> {
  return request<MeetingJob>(`/meetings/${jobId}/retry`, { method: "POST" });
}

/**
 * Get availability + default-on status for opt-in features
 * (LLM correction, polish pass, WhisperX). Used by the upload UI to
 * gate toggles behind whether the underlying dependency is installed.
 */
export async function getSystemFeatures(): Promise<SystemFeatures> {
  return request<SystemFeatures>("/system/features");
}
