/** Pipeline processing state matching backend PipelineState enum. */
export type PipelineState =
  | "uploaded"
  | "preprocessing"
  | "transcribing"
  | "diarizing"
  | "aligning"
  | "summarizing"
  | "qa_validating"
  | "formatting"
  | "completed"
  | "error"
  | "retrying";

/**
 * Meeting type — matches backend MeetingType enum in backend/app/models.py.
 * Drives prompt selection and DOCX template. ASR routing is handled
 * separately by language detection unless the user overrides it.
 */
export type MeetingType =
  | "court_hearing"
  | "administrative"
  | "client_meeting"
  | "interview"
  | "generic";

/** Human-facing Russian labels for each MeetingType, in UI order. */
export const MEETING_TYPE_LABELS: Record<MeetingType, string> = {
  court_hearing: "Судебное заседание",
  administrative: "Административное совещание",
  client_meeting: "Встреча с клиентом",
  interview: "Интервью",
  generic: "Общая встреча",
};

/** UI-order list for dropdown rendering. */
export const MEETING_TYPES_IN_ORDER: MeetingType[] = [
  "generic",
  "court_hearing",
  "administrative",
  "client_meeting",
  "interview",
];

/** ASR engine options surfaced to the UI. Backend supports more
 * (qwen, nemo, hf-whisper) but those are not user-facing in the
 * advanced section — they're for power-user .env overrides only.
 */
export type AsrEngineChoice =
  | "auto"
  | "gigaam"
  | "whisper"
  | "whisperx";

export const ASR_ENGINE_LABELS: Record<AsrEngineChoice, string> = {
  auto: "Авто (рекомендуется)",
  gigaam: "GigaAM v3 (русский)",
  whisper: "Whisper (мульти-язычный)",
  whisperx: "WhisperX (экспериментально)",
};

export const ASR_ENGINE_DESCRIPTIONS: Record<AsrEngineChoice, string> = {
  auto: "Определяет язык записи и выбирает лучший движок автоматически.",
  gigaam:
    "Лучшая точность на русском (Sber, 700K часов обучения). " +
    "Рекомендуется для всех русскоязычных совещаний и заседаний.",
  whisper:
    "Универсальный многоязычный движок (OpenAI Whisper large-v3). " +
    "Имеет смысл для записей с заметной долей английского.",
  whisperx:
    "Whisper + фонетическое выравнивание. На бенчмарке court_hearing_129 " +
    "показал WER 32.5% против 25.7% у GigaAM (на 7 п.п. хуже). " +
    "Использовать только если критичны точные таймкоды на каждое слово.",
};

/** /api/system/features response shape. */
export interface SystemFeature {
  available: boolean;
  default_enabled: boolean;
  label_ru: string;
  description_ru: string;
  install_hint_ru?: string;
}

export interface SystemFeatures {
  llm_correction: SystemFeature;
  polish_pass: SystemFeature;
  whisperx: SystemFeature;
  normalize_numbers: SystemFeature;
  cloud_compare: SystemFeature;
}

/**
 * Speaker-count buckets for the upload UI (T3.1, 2026-04-23).
 *
 * Tight ranges around an expected count help pyannote's VBx clustering
 * converge correctly — confirmed by EPAM IT team's setup which uses
 * `--min-speakers 16 --max-speakers 20` for their admin meetings.
 *
 * Each bucket maps to (min, max) bounds the orchestrator passes
 * directly to pyannote. "auto" sends nothing — pyannote uses its
 * default broad range.
 */
export type SpeakerBucket =
  | "auto"
  | "2"
  | "3-5"
  | "6-10"
  | "11-15"
  | "16-20"
  | "21+";

export const SPEAKER_BUCKETS_IN_ORDER: SpeakerBucket[] = [
  "auto",
  "2",
  "3-5",
  "6-10",
  "11-15",
  "16-20",
  "21+",
];

export const SPEAKER_BUCKET_LABELS: Record<SpeakerBucket, string> = {
  auto: "Auto-detect",
  "2": "2 speakers",
  "3-5": "3–5 speakers",
  "6-10": "6–10 speakers",
  "11-15": "11–15 speakers",
  "16-20": "16–20 speakers",
  "21+": "More than 20",
};

/**
 * Resolve a bucket to (min, max) for the backend. Returns null for
 * "auto" — caller omits the form fields entirely so the orchestrator
 * uses config defaults.
 */
export function resolveSpeakerBucket(
  bucket: SpeakerBucket,
): { min: number; max: number } | null {
  switch (bucket) {
    case "auto":
      return null;
    case "2":
      return { min: 2, max: 2 };
    case "3-5":
      return { min: 3, max: 5 };
    case "6-10":
      return { min: 6, max: 10 };
    case "11-15":
      return { min: 11, max: 15 };
    case "16-20":
      return { min: 16, max: 20 };
    case "21+":
      return { min: 20, max: 40 };
  }
}

/** Meeting processing job. */
export interface MeetingJob {
  id: string;
  filename: string;
  state: PipelineState;
  created_at: string;
  updated_at: string;
  progress: number;
  current_stage: string;
  eta_seconds: number | null;
  error: string | null;
  retry_count: number;
  context: string;
  meeting_type: MeetingType;
  owner_username?: string | null;
  expected_speakers_min?: number | null;
  expected_speakers_max?: number | null;
  court_participants?: string;
  court_dictionary?: string;
}

/** Aligned transcript segment with speaker info. */
export interface AlignedSegment {
  start: number;
  end: number;
  text: string;
  speaker_id: string;
  speaker_name: string | null;
  suggested_speaker_name?: string | null;
  suggested_speaker_confidence?: number | null;
  /** Word-level ASR confidence (0..1). */
  confidence: number;
  /**
   * Speaker-attribution confidence (0..1) — what fraction of the
   * segment's diarized time was assigned to the chosen speaker.
   * Low values mean the segment is contested (multiple speakers
   * overlapping, or very brief turn). null when alignment had no
   * diarization to work with. Used by the UI to surface segments
   * that need a human review.
   */
  attribution_confidence: number | null;
}

export interface TranscriptReviewSpeaker {
  speaker_id: string;
  speaker_name: string | null;
  suggested_speaker_name?: string | null;
  suggested_speaker_confidence?: number | null;
  turn_count: number;
  duration_s: number;
  low_confidence_turns: number;
  samples: string[];
}

export interface TranscriptReviewDictionaryTerm {
  term: string;
  count: number;
  found: boolean;
}

export interface TranscriptReview {
  job_id: string;
  meeting_type: MeetingType | null;
  expected_speakers_min: number | null;
  expected_speakers_max: number | null;
  detected_speakers: number;
  speaker_count_matches: boolean | null;
  court_participants: string;
  dictionary_terms: TranscriptReviewDictionaryTerm[];
  dictionary_missing_count: number;
  low_confidence_turns: number;
  speaker_review?: {
    confirmed?: boolean;
    confirmed_at?: string;
    confirmed_by?: string;
    invalidated_at?: string;
    invalidated_reason?: string;
  };
  speakers: TranscriptReviewSpeaker[];
}

/** Meeting participant. */
export interface Participant {
  speaker_id: string;
  speaker_name: string | null;
  speaking_time: number;
  speaking_share: number;
}

/** Discussion topic. */
export interface TopicItem {
  title: string;
  content: string;
  speakers: string[];
}

/** Meeting decision. */
export interface Decision {
  text: string;
  responsible: string | null;
}

/** Action item from meeting. */
export interface TaskItem {
  text: string;
  assignee: string | null;
  deadline: string | null;
}

/** Complete meeting protocol (legacy GENERIC schema).
 *
 * Used for meeting_type = "generic" or older protocols on disk that
 * predate the wrapped {meeting_type, payload} shape (Block 6, T3.3).
 * Newer admin/court/client/interview protocols come through the
 * `WrappedProtocol` types below.
 */
export interface MeetingProtocol {
  meeting_date: string | null;
  topic: string;
  participants: Participant[];
  summary: string;
  key_topics: TopicItem[];
  decisions: Decision[];
  tasks: TaskItem[];
  open_questions: string[];
  transcript: AlignedSegment[];
}

// ---------------------------------------------------------------------
// Wrapped protocol payloads (Block 6 onward).
//
// Backend's GET /protocol/data returns
//   {meeting_type: "<type>", payload: {...}}
// for all non-generic meeting types. Each payload type has its own
// shape — frontend dispatches on meeting_type to render the right view.
// Schemas mirror backend/engine/protocols/schemas.py.
// ---------------------------------------------------------------------

/** One verbatim turn in a stenogram (court_hearing or admin). */
export interface StenogramTurn {
  speaker: string;
  text: string;
  /** Optional segment start time in seconds. Renders as [HH:MM:SS]. */
  start_s: number | null;
}

/** Flat 9-section admin item: decision/task/open_question/risk.
 *
 * Sprint 2026-05-04 (task #45): added `confidence` so the UI can
 * surface low-confidence items with an amber "[?]" tag. The model
 * marks each extracted item as high/medium/low in the primary
 * extraction call (replacing the previous defensive verification
 * pass that deleted items without evidence).
 */
export interface FlatProtocolItem {
  /** Practical admin item type. Older protocols may omit it. */
  kind?: "decision" | "task" | "open_question" | "risk" | "thesis";
  text: string;
  /** Who raised/decided this. Empty allowed; user fills in post-hoc. */
  speaker: string;
  /** Short transcript-grounded evidence phrase. */
  evidence: string;
  /** Optional timestamp near the evidence quote. */
  timestamp?: string;
  /** Department/person executing the task. Tasks-only. */
  owner: string | null;
  /** DD.MM.YYYY or short free text. Tasks-only. */
  deadline: string | null;
  /** LLM-assigned confidence. Older protocols without this field
   * default to "medium" on the backend so they render reasonably. */
  confidence?: "high" | "medium" | "low";
}

/** Court-hearing protocol — stenogram + summary header. */
export interface CourtHearingPayload {
  title: string;
  case_number: string;
  hearing_date: string;
  summary: string;
  participants: string[];
  turns: StenogramTurn[];
}

/** One topic block in the map-reduce admin layout (Sprint 2026-04-30). */
export interface TopicSummary {
  /** Short topic name in Russian, e.g. "Подбор персонала и HR". */
  name: string;
  /** 1-3 sentences summarising the discussion under this topic. */
  discussion: string;
  /** Topic-scoped decisions; empty when none were raised. */
  decisions: FlatProtocolItem[];
  /** Topic-scoped tasks. */
  tasks: FlatProtocolItem[];
  /** Topic-scoped open questions. */
  open_questions: FlatProtocolItem[];
}

/** Administrative protocol — supports flat (T3.3) + topic-segmented (Sprint 2026-04-30). */
export interface AdministrativePayload {
  title: string;
  meeting_date: string;
  meeting_goal: string;
  summary: string;
  participants: string[];
  topics: string[];
  /** Practical v1.0 admin layout: unified useful takeaways. */
  items?: FlatProtocolItem[];
  decisions: FlatProtocolItem[];
  tasks: FlatProtocolItem[];
  open_questions: FlatProtocolItem[];
  risks: FlatProtocolItem[];
  /** Map-reduce per-topic blocks. When non-empty, the UI renders a
   * topic-grouped layout; when empty, falls back to flat sections. */
  topic_summaries: TopicSummary[];
  /** Legacy department-grouped layout — empty for new protocols. */
  departments: unknown[];
  turns: StenogramTurn[];
}

/** Client meeting commitment — either side, optional deadline. */
export interface ClientCommitment {
  text: string;
  party: "epam" | "client";
  speaker: string;
  deadline: string | null;
}

/** Item with speaker attribution (used in client meeting topics/requests/follow-ups). */
export interface SpeakerAttributedItem {
  text: string;
  speaker: string;
  deadline: string | null;
}

/** Client meeting protocol. */
export interface ClientMeetingPayload {
  title: string;
  meeting_date: string;
  client_name: string;
  epam_representatives: string[];
  client_representatives: string[];
  summary: string;
  topics: SpeakerAttributedItem[];
  client_requests: SpeakerAttributedItem[];
  commitments: ClientCommitment[];
  follow_ups: SpeakerAttributedItem[];
  parked_questions: string[];
}

/** Interview observation with verbatim candidate quote. */
export interface InterviewObservation {
  observation: string;
  evidence: string;
}

/** Internal (confidential) interview scorecard. */
export interface InternalInterviewReport {
  role: string;
  candidate_name: string;
  summary: string;
  strengths: InterviewObservation[];
  concerns: InterviewObservation[];
  technical_signals: InterviewObservation[];
  communication_signals: InterviewObservation[];
  open_questions_for_next_round: string[];
}

/** External (candidate-facing) interview report. */
export interface ExternalInterviewReport {
  candidate_name: string;
  speaking_style_notes: string[];
  strongest_moments: InterviewObservation[];
  areas_to_develop: string[];
  closing_note: string;
}

/** Interview protocol — both reports. */
export interface InterviewPayload {
  meeting_date: string;
  internal: InternalInterviewReport;
  external: ExternalInterviewReport;
}

/** Wrapped protocol returned by GET /protocol/data for non-generic types. */
export type WrappedProtocol =
  | { meeting_type: "court_hearing"; payload: CourtHearingPayload }
  | { meeting_type: "administrative"; payload: AdministrativePayload }
  | { meeting_type: "client_meeting"; payload: ClientMeetingPayload }
  | { meeting_type: "interview"; payload: InterviewPayload };

/**
 * Type guard: is this protocol response in the wrapped shape?
 *
 * Used by ProtocolPage to dispatch between legacy (MeetingProtocol)
 * and wrapped (WrappedProtocol) rendering paths. Wrapped responses
 * always carry both `meeting_type` (string) and `payload` (object).
 */
export function isWrappedProtocol(
  data: MeetingProtocol | WrappedProtocol | null | undefined,
): data is WrappedProtocol {
  if (!data) return false;
  const obj = data as Record<string, unknown>;
  return (
    typeof obj.meeting_type === "string" &&
    typeof obj.payload === "object" &&
    obj.payload !== null
  );
}

/** Voice profile (Block 7, Sprint 2026-04-30).
 *
 * Replaces the legacy file-based "Speaker" model — embeddings now
 * live in encrypted SQLite at the shared data root, not in a JSON
 * file. New fields: department, sample_count, match_count,
 * last_used_at. Legacy fields (embedding_path, meetings_count)
 * retained for backward-compat with old protocols on disk.
 */
export interface SpeakerProfile {
  id: string;
  name: string;
  position: string | null;
  department: string;
  is_admin_default: boolean;
  embedding_path: string;
  registered_at: string;
  last_used_at: string | null;
  meetings_count: number;
  sample_count: number;
  match_count: number;
}

/** System health response. */
export interface SystemHealth {
  status: string;
  cpu_percent: number;
  memory_percent: number;
  memory_available_gb: number;
  active_jobs: number;
}

/** GPU status response. */
export interface GpuStatus {
  vram_used_gb: number;
  vram_total_gb: number;
  vram_free_gb: number;
  gpu_utilization: number;
  temperature: number;
  device_name: string;
  current_model: string | null;
  is_available: boolean;
  source?: string;
  is_dummy?: boolean;
}

/** System stats response. */
export interface SystemStats {
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  processing_jobs: number;
  pending_jobs: number;
}

/** WebSocket progress message. */
export interface ProgressMessage {
  type: "progress" | "status" | "pong" | "error";
  job_id: string;
  progress: number;
  state: PipelineState;
  current_stage: string;
  eta_seconds: number | null;
  timestamp: string;
}

/** JWT auth response. */
export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in?: number;
  must_change_password?: boolean;
  username?: string;
  role?: string;
}

export interface CurrentUser {
  username: string;
  role: string;
  must_change_password: boolean;
}

export interface UserAccount {
  username: string;
  role: "admin" | "operator" | "viewer" | "lawyer" | "developer" | string;
  is_active: boolean;
  must_change_password: boolean;
  created_at: string;
  updated_at: string;
}

export type DeveloperModelCategory = "asr" | "diarization" | "llm";

export interface DeveloperModelEntry {
  id: string;
  name: string;
  provider: string;
  status: string;
  engine?: string;
  model?: string;
  base_url?: string;
  context_tokens?: number;
  structured_outputs?: boolean;
  recommended_for?: string[];
  notes?: string;
  [key: string]: unknown;
}

export interface DeveloperPrompt {
  id: string;
  meeting_type: string;
  stage: string;
  title: string;
  enabled: boolean;
  version: number;
  system_template: string;
  user_template: string;
  placeholders: string[];
  notes: string;
  updated_at?: string;
  updated_by?: string;
}

export interface DeveloperSettings {
  version: number;
  updated_at: string;
  active: {
    profile: string;
    asr: string;
    diarization: string;
    llm: string;
  };
  profiles: Array<{
    id: string;
    name: string;
    description: string;
  }>;
  catalog: Record<DeveloperModelCategory, DeveloperModelEntry[]>;
  prompts: DeveloperPrompt[];
}
