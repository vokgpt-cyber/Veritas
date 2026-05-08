/**
 * Audio upload page with drag-and-drop, meeting context, and file attachments.
 *
 * After successful upload, redirects to the processing page
 * to show real-time progress via WebSocket.
 */

import { useState, useEffect, useRef, useCallback, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  FileAudio,
  FileText,
  X,
  AlertCircle,
  Loader2,
  CheckCircle,
  Paperclip,
  Users,
  BookOpen,
} from "lucide-react";
import { uploadAudio, getGpuStatus, listSpeakers } from "../api/client";
import GpuStatus from "../components/GpuStatus";
import type {
  GpuStatus as GpuStatusType,
  SpeakerProfile,
} from "../types/api";
import {
  MEETING_TYPE_LABELS,
  MEETING_TYPES_IN_ORDER,
  type MeetingType,
  type SpeakerBucket,
  SPEAKER_BUCKETS_IN_ORDER,
  SPEAKER_BUCKET_LABELS,
  resolveSpeakerBucket,
} from "../types/api";
import { usePreferences } from "../context/PreferencesContext";

// Accepted media extensions. Audio formats decoded directly; video
// formats go through ffmpeg's demuxer to extract the audio track
// (see backend/core/audio.py). Court hearings are often delivered
// as MP4 — added 2026-04-23 per user request.
const ALLOWED_AUDIO_EXT = [
  // Audio
  ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
  // Video (audio track extracted via ffmpeg)
  ".mp4", ".mov", ".avi", ".mkv", ".webm",
];
const ALLOWED_CONTEXT_EXT = [".docx", ".xlsx", ".xls", ".pdf", ".txt"];
const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2 GB

const UPLOAD_COPY = {
  en: {
    title: "Upload Meeting Recording",
    subtitle:
      "Upload an audio file to transcribe and generate a meeting protocol. Supports audio (MP3, WAV, M4A, FLAC, OGG, AAC) and video (MP4, MOV, AVI, MKV, WEBM); audio track is extracted automatically.",
    dropActive: "Drop your file here",
    dropIdle: "Drag and drop an audio file here",
    browse: "or click to browse",
    removeFile: "Remove file",
    meetingType: "Meeting type",
    meetingTypeHelp:
      "Selects the protocol template. ASR is routed by detected language: Russian-primary recordings use GigaAM v3; English-heavy recordings use Whisper.",
    adminAttendees: "Meeting participants",
    recommended: "recommended",
    adminAttendeesHelp:
      "Select actual attendees for context/name mapping. This list does not set pyannote speaker count; use Expected number of speakers below.",
    customAttendees: "Other participants",
    commaSeparated: "comma-separated",
    customAttendeesPlaceholder: "For example: I. Petrov, M. Sidorova",
    courtExactSpeakers: "Exact speaker count",
    courtExactHelp:
      "For court hearings this is sent as min=max to pyannote. Leave empty only when the count is genuinely unknown.",
    courtParticipants: "Participants and roles",
    courtParticipantsPlaceholder:
      "Court: judge. Plaintiff: name/role. Defendant: name/role. Witnesses/experts: names if known.",
    courtDictionary: "Case dictionary",
    courtDictionaryPlaceholder:
      "Names, company names, case numbers, addresses, amounts, contract numbers, rare legal terms. One item per line or separated by commas.",
    courtDictionaryHelp:
      "The dictionary is used as review context and for highlighting suspicious/missing terms. It does not silently rewrite the transcript.",
    context: "Meeting context",
    optional: "optional",
    contextPlaceholder:
      "E.g.: Administrative meeting reviewing Q1 tasks. Participants: Ivanov (partner), Petrov (associate). Case #2024-1187.",
    contextHelp:
      "Describe the meeting type, participants, and topic. This helps the AI produce a more accurate summary.",
    documents: "Supporting documents",
    attachDocuments: "Attach documents (DOCX, XLSX, PDF, TXT)",
    documentsHelp:
      "Text from attached files will be provided to the AI as additional context.",
    expectedSpeakers: "Expected number of speakers",
    expectedSpeakersHelp:
      "Picking a tight range significantly improves speaker identification. Use Auto-detect only when you genuinely do not know the count.",
    advanced: "Advanced options",
    advancedClosed: "quality, ASR engine",
    llmOptional: "optional",
    courtLlmWarning:
      "Court transcripts are legal artifacts. Keep this off unless you want an LLM-suggested correction pass; the original pre-correction transcript is saved for audit.",
    polishAdminOnly: "Available only for Administrative meeting.",
    asrEngine: "Recognition engine",
    whisperxInstall: "Not installed. Run",
    vramLowTitle:
      "Not enough free GPU memory. Close other GPU applications and try again.",
    vramLow:
      "Button is locked: GPU is busy. Free video memory and refresh the GPU indicator above.",
    uploading: "Uploading...",
    start: "Start Processing",
    security:
      "All processing is performed on-premise. Your audio files and documents never leave this machine. Files are automatically deleted after protocol generation.",
    securityLabel: "Security:",
    exactError: "Exact speaker count must be between 1 and 40.",
  },
  ru: {
    title: "Загрузка записи",
    subtitle:
      "Загрузите аудио или видео: VERITAS сделает стенограмму и подготовит протокол. Поддерживаются аудио (MP3, WAV, M4A, FLAC, OGG, AAC) и видео (MP4, MOV, AVI, MKV, WEBM); звуковая дорожка извлекается автоматически.",
    dropActive: "Отпустите файл здесь",
    dropIdle: "Перетащите аудио или видео сюда",
    browse: "или нажмите, чтобы выбрать файл",
    removeFile: "Удалить файл",
    meetingType: "Тип материала",
    meetingTypeHelp:
      "Выбирает шаблон протокола. Распознавание выбирается по языку записи: для русской речи используется GigaAM v3, для записей с заметной долей английского — Whisper.",
    adminAttendees: "Участники совещания",
    recommended: "рекомендуется",
    adminAttendeesHelp:
      "Отметьте присутствующих для контекста и сопоставления имен. Этот список не задает число активных спикеров для pyannote; ниже отдельно укажите ожидаемых говорящих.",
    customAttendees: "Другие участники",
    commaSeparated: "через запятую",
    customAttendeesPlaceholder: "Например: И. Петров, М. Сидорова",
    courtExactSpeakers: "Точное число спикеров",
    courtExactHelp:
      "Для судебных заседаний это передается в pyannote как min=max. Оставляйте пустым только если число действительно неизвестно.",
    courtParticipants: "Участники и роли",
    courtParticipantsPlaceholder:
      "Суд: судья. Истец: имя/роль. Ответчик: имя/роль. Свидетели/эксперты: имена, если известны.",
    courtDictionary: "Словарь дела",
    courtDictionaryPlaceholder:
      "Фамилии, компании, номера дел, адреса, суммы, номера договоров, редкие юридические термины. По одному на строку или через запятую.",
    courtDictionaryHelp:
      "Словарь используется для контекста и проверки спорных/пропущенных терминов. Он не переписывает стенограмму автоматически.",
    context: "Контекст встречи",
    optional: "необязательно",
    contextPlaceholder:
      "Например: административное совещание по задачам Q1. Участники: Иванов (партнер), Петров (юрист). Дело №2024-1187.",
    contextHelp:
      "Опишите тип встречи, участников и тему. Это помогает ИИ точнее подготовить протокол.",
    documents: "Поддерживающие документы",
    attachDocuments: "Прикрепить документы (DOCX, XLSX, PDF, TXT)",
    documentsHelp:
      "Текст из приложенных файлов будет передан ИИ как дополнительный контекст.",
    expectedSpeakers: "Ожидаемое число спикеров",
    expectedSpeakersHelp:
      "Узкий диапазон заметно улучшает определение спикеров. Используйте автоопределение только когда число действительно неизвестно.",
    advanced: "Дополнительные опции",
    advancedClosed: "качество, движок ASR",
    llmOptional: "опционально",
    courtLlmWarning:
      "Судебная стенограмма — юридический документ. Лучше держать эту опцию выключенной, если не нужен отдельный LLM-проход с предлагаемыми исправлениями; исходная версия сохраняется для аудита.",
    polishAdminOnly: "Доступно только для административных совещаний.",
    asrEngine: "Движок распознавания",
    whisperxInstall: "Не установлен. Запустите",
    vramLowTitle:
      "Недостаточно свободной видеопамяти. Закройте другие GPU-приложения и попробуйте снова.",
    vramLow:
      "Кнопка заблокирована: GPU занят. Освободите видеопамять и обновите индикатор выше.",
    uploading: "Загрузка...",
    start: "Начать обработку",
    security:
      "Вся обработка выполняется локально. Аудио, видео и документы не покидают этот компьютер. Временные файлы автоматически удаляются после подготовки протокола.",
    securityLabel: "Безопасность:",
    exactError: "Точное число спикеров должно быть от 1 до 40.",
  },
} as const;

const MEETING_LABELS = {
  en: {
    generic: "Generic meeting",
    court_hearing: "Court hearing",
    administrative: "Administrative meeting",
    client_meeting: "Client meeting",
    interview: "Interview",
  },
  ru: {
    generic: "Общая встреча",
    court_hearing: "Судебное заседание",
    administrative: "Административное совещание",
    client_meeting: "Встреча с клиентом",
    interview: "Интервью",
  },
} as const;

const SPEAKER_BUCKET_COPY = {
  en: SPEAKER_BUCKET_LABELS,
  ru: {
    auto: "Автоопределение",
    "2": "2 спикера",
    "3-5": "3-5 спикеров",
    "6-10": "6-10 спикеров",
    "11-15": "11-15 спикеров",
    "16-20": "16-20 спикеров",
    "21+": "Больше 20",
  },
} as const;

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getExtension(filename: string): string {
  const idx = filename.lastIndexOf(".");
  return idx >= 0 ? filename.slice(idx).toLowerCase() : "";
}

export default function UploadPage() {
  const navigate = useNavigate();
  const { language, resolvedTheme } = usePreferences();
  const copy = UPLOAD_COPY[language];
  const isDark = resolvedTheme === "dark";
  const fileInputRef = useRef<HTMLInputElement>(null);
  const contextFileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // T3.1 (2026-04-23): replaced numeric expected-speakers dropdown
  // with named buckets matching pyannote tight-range guidance from
  // EPAM IT. "auto" sends no constraint; everything else maps to
  // (min, max) via resolveSpeakerBucket().
  const [speakerBucket, setSpeakerBucket] = useState<SpeakerBucket>("auto");
  const [context, setContext] = useState("");
  const [contextFiles, setContextFiles] = useState<File[]>([]);
  const [courtExactSpeakers, setCourtExactSpeakers] = useState("");
  const [courtParticipants, setCourtParticipants] = useState("");
  const [courtDictionary, setCourtDictionary] = useState("");
  // Meeting type drives prompt and DOCX template. ASR routing is
  // handled by backend policy so operators do not need low-level knobs.
  const [meetingType, setMeetingType] = useState<MeetingType>("generic");

  // Administrative attendance picker — only visible when
  // meetingType === "administrative". Selected IDs get prepended to
  // the context at submit time so the LLM's `participants` extraction
  // has a strong hint. This is intentionally separate from the
  // expected active-speaker bucket: many attendees never speak.
  const [employeeProfiles, setEmployeeProfiles] = useState<SpeakerProfile[]>([]);
  const [employeesLoading, setEmployeesLoading] = useState(false);
  const [selectedAdminAttendees, setSelectedAdminAttendees] = useState<
    Set<string>
  >(new Set());
  const [customAttendees, setCustomAttendees] = useState("");

  // VRAM pre-flight: poll GPU status so we can grey out the upload
  // button when memory is too low to start the pipeline. Without
  // this the user uploads, waits 1-2 min for preprocessing +
  // language detection, then sees a cryptic mid-pipeline VRAM error.
  // Threshold: 4 GB matches the backend pre-flight (smallest ASR
  // engine, GigaAM, needs ~3 GB; we leave 1 GB margin).
  const [gpu, setGpu] = useState<GpuStatusType | null>(null);
  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const s = await getGpuStatus();
        if (!cancelled) setGpu(s);
      } catch {
        /* ignore */
      }
    }
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (meetingType === "administrative") {
      setSpeakerBucket((current) => (current === "auto" ? "11-15" : current));
    }
  }, [meetingType]);
  const vramLow =
    gpu !== null && gpu.is_available && gpu.vram_free_gb < 4.0;

  // Load the shared employee directory. Default admin attendees are
  // preselected, and the operator can adjust the actual attendance.
  useEffect(() => {
    let cancelled = false;
    setEmployeesLoading(true);
    listSpeakers()
      .then((profiles) => {
        if (cancelled) return;
        setEmployeeProfiles(profiles);
        setSelectedAdminAttendees((prev) => {
          if (prev.size > 0) return prev;
          return new Set(
            profiles
              .filter((p) => p.is_admin_default)
              .map((p) => p.id),
          );
        });
      })
      .catch(() => {
        if (!cancelled) setEmployeeProfiles([]);
      })
      .finally(() => {
        if (!cancelled) setEmployeesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const validateFile = useCallback((f: File): string | null => {
    const ext = getExtension(f.name);
    if (!ALLOWED_AUDIO_EXT.includes(ext)) {
      return `Unsupported format "${ext}". Allowed: ${ALLOWED_AUDIO_EXT.join(", ")}`;
    }
    if (f.size > MAX_FILE_SIZE) {
      return `File too large (${formatFileSize(f.size)}). Maximum: 2 GB`;
    }
    if (f.size === 0) {
      return "File is empty";
    }
    return null;
  }, []);

  const handleFileSelect = useCallback(
    (f: File) => {
      setError(null);
      const err = validateFile(f);
      if (err) {
        setError(err);
        return;
      }
      setFile(f);
    },
    [validateFile],
  );

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (selected) {
      handleFileSelect(selected);
    }
  }

  function handleRemoveFile() {
    setFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function handleContextFileAdd(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files) return;
    const newFiles: File[] = [];
    for (let i = 0; i < files.length; i++) {
      const ext = getExtension(files[i].name);
      if (ALLOWED_CONTEXT_EXT.includes(ext)) {
        newFiles.push(files[i]);
      }
    }
    setContextFiles((prev) => [...prev, ...newFiles]);
    if (contextFileInputRef.current) {
      contextFileInputRef.current.value = "";
    }
  }

  function removeContextFile(idx: number) {
    setContextFiles((prev) => prev.filter((_, i) => i !== idx));
  }

  /**
   * Build the final context string sent to the backend. For admin
   * meetings, prepend an attendee line built from the employee
   * directory and any custom names the user typed.
   */
  function parseCustomAttendees(): string[] {
    return customAttendees
      .split(/[,;\n]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }

  function formatEmployeeForContext(profile: SpeakerProfile): string {
    const meta = [profile.department, profile.position].filter(Boolean);
    return meta.length > 0
      ? `${profile.name} (${meta.join(", ")})`
      : profile.name;
  }

  function getSelectedAdminProfiles(): SpeakerProfile[] {
    return employeeProfiles.filter((p) => selectedAdminAttendees.has(p.id));
  }

  function buildSubmitContext(): string | undefined {
    const userContext = (context || "").trim();

    if (meetingType !== "administrative") {
      return userContext || undefined;
    }

    const checked = getSelectedAdminProfiles().map(formatEmployeeForContext);
    const custom = parseCustomAttendees();

    const allAttendees = [...checked, ...custom];

    if (allAttendees.length === 0) {
      return userContext || undefined;
    }

    const hint = `Участники совещания: ${allAttendees.join(", ")}.`;
    const combined = userContext
      ? `${hint}\n\n${userContext}`
      : hint;
    return combined;
  }

  async function handleUpload() {
    if (!file) return;

    setError(null);
    setUploading(true);

    try {
      let speakerRange = resolveSpeakerBucket(speakerBucket) ?? undefined;
      const exactSpeakerText = courtExactSpeakers.trim();
      if (meetingType === "court_hearing") {
        speakerRange = undefined;
        if (exactSpeakerText) {
          const exact = Number.parseInt(exactSpeakerText, 10);
          if (!Number.isFinite(exact) || exact < 1 || exact > 40) {
            throw new Error(copy.exactError);
          }
          speakerRange = { min: exact, max: exact };
        }
      }
      const job = await uploadAudio(file, {
        meetingType,
        speakerRange,
        context: buildSubmitContext(),
        contextFiles: contextFiles.length > 0 ? contextFiles : undefined,
        courtParticipants:
          meetingType === "court_hearing"
            ? courtParticipants.trim() || undefined
            : undefined,
        courtDictionary:
          meetingType === "court_hearing"
            ? courtDictionary.trim() || undefined
            : undefined,
      });
      navigate(`/processing/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  }

  /** Toggle a single attendee checkbox. */
  function toggleAdminAttendee(id: string) {
    setSelectedAdminAttendees((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  const dropZoneStateClass = dragging
    ? "border-epam-red bg-epam-red/5"
    : file
      ? isDark
        ? "border-emerald-500 bg-epam-gray-100"
        : "border-green-300 bg-green-50"
      : "border-epam-gray-300 bg-white hover:border-epam-gray-400 hover:bg-epam-gray-50";
  const adminEmployeeOptions = [...employeeProfiles].sort((a, b) => {
    if (a.is_admin_default !== b.is_admin_default) {
      return a.is_admin_default ? -1 : 1;
    }
    return a.name.localeCompare(b.name, "ru");
  });

  return (
    <div className="max-w-2xl mx-auto px-6 py-10">
      <h1 className="font-heading text-2xl text-epam-gray-900 mb-2">
        {copy.title}
      </h1>
      <p className="text-sm text-epam-gray-500 mb-8">
        {copy.subtitle}
      </p>

      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !file && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer ${dropZoneStateClass}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_AUDIO_EXT.join(",")}
          onChange={handleInputChange}
          className="hidden"
        />

        {!file ? (
          <>
            <Upload
              size={40}
              className={`mx-auto mb-4 ${dragging ? "text-epam-red" : "text-epam-gray-400"}`}
            />
            <p className="text-sm font-medium text-epam-gray-700 mb-1">
              {dragging
                ? copy.dropActive
                : copy.dropIdle}
            </p>
            <p className="text-xs text-epam-gray-400">
              {copy.browse} &middot; audio: MP3, WAV, M4A, FLAC, OGG, AAC &middot; video: MP4, MOV, AVI, MKV, WEBM
            </p>
          </>
        ) : (
          <div className="flex items-center justify-center gap-4">
            <FileAudio
              size={32}
              className={`shrink-0 ${isDark ? "text-emerald-300" : "text-green-600"}`}
            />
            <div className="text-left min-w-0">
              <p className="text-sm font-medium text-epam-gray-900 truncate">
                {file.name}
              </p>
              <p className="text-xs text-epam-gray-500">
                {formatFileSize(file.size)}
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleRemoveFile();
              }}
              className="p-1 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
              title={copy.removeFile}
            >
              <X size={18} />
            </button>
          </div>
        )}
      </div>

      {/* Options panel — shown after file selected */}
      {file && (
        <div className="mt-6 space-y-5">
          {/* Meeting type — drives prompt + DOCX template + ASR routing */}
          <div>
            <label
              htmlFor="meeting-type"
              className="block text-sm font-medium text-epam-gray-700 mb-1"
            >
              {copy.meetingType}
            </label>
            <select
              id="meeting-type"
              value={meetingType}
              onChange={(e) => setMeetingType(e.target.value as MeetingType)}
              className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red"
            >
              {MEETING_TYPES_IN_ORDER.map((mt) => (
                <option key={mt} value={mt}>
                  {MEETING_LABELS[language][mt] || MEETING_TYPE_LABELS[mt]}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-epam-gray-400">
              {copy.meetingTypeHelp}
            </p>
          </div>

          {/* Admin attendees — shown only for administrative meetings */}
          {meetingType === "administrative" && (
            <div>
              <label className="block text-sm font-medium text-epam-gray-700 mb-1">
                {copy.adminAttendees}{" "}
                <span className="text-epam-gray-400 font-normal">({copy.recommended})</span>
              </label>
              <p className="text-xs text-epam-gray-500 mb-2">
                {copy.adminAttendeesHelp}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-y-1.5 gap-x-4 mb-3">
                {employeesLoading && (
                  <p className="text-xs text-epam-gray-400">
                    {language === "ru"
                      ? "Загружаю справочник сотрудников..."
                      : "Loading employee directory..."}
                  </p>
                )}
                {!employeesLoading && employeeProfiles.length === 0 && (
                  <p className="text-xs text-epam-gray-400 sm:col-span-2">
                    {language === "ru"
                      ? "Справочник сотрудников пока пуст. Добавьте сотрудников в разделе \"Голоса\"."
                      : "The employee directory is empty. Add employees in Voices first."}
                  </p>
                )}
                {adminEmployeeOptions.map((p) => (
                  <label
                    key={p.id}
                    className="flex items-start gap-2 cursor-pointer text-sm group"
                  >
                    <input
                      type="checkbox"
                      checked={selectedAdminAttendees.has(p.id)}
                      onChange={() => toggleAdminAttendee(p.id)}
                      className="mt-0.5 h-4 w-4 accent-epam-red cursor-pointer"
                    />
                    <span className="min-w-0">
                      <span className="font-medium text-epam-gray-800">
                        {p.name}
                      </span>
                      {p.is_admin_default && (
                        <span className="ml-2 text-[10px] uppercase tracking-wide text-epam-red">
                          {language === "ru" ? "обычно" : "default"}
                        </span>
                      )}
                      <span className="block text-xs text-epam-gray-400">
                        {[p.department, p.position].filter(Boolean).join(", ")}
                        {p.sample_count > 0
                          ? language === "ru"
                            ? " · голос есть"
                            : " · voice sample"
                          : language === "ru"
                            ? " · без голоса"
                            : " · no voice"}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
              <label
                htmlFor="custom-attendees"
                className="block text-xs font-medium text-epam-gray-600 mb-1"
              >
                {copy.customAttendees}{" "}
                <span className="text-epam-gray-400 font-normal">
                  ({copy.commaSeparated})
                </span>
              </label>
              <input
                id="custom-attendees"
                type="text"
                value={customAttendees}
                onChange={(e) => setCustomAttendees(e.target.value)}
                placeholder={copy.customAttendeesPlaceholder}
                className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red placeholder:text-epam-gray-300"
              />
              <p className="mt-1 text-xs text-epam-gray-400">
                {language === "ru"
                  ? "Список по умолчанию настраивается в разделе \"Голоса\". Число активных спикеров для pyannote задается отдельным полем ниже."
                  : "The default list is configured in Voices. pyannote active-speaker count is set separately below."}
              </p>
            </div>
          )}

          {meetingType === "court_hearing" && (
            <div className="border border-epam-gray-200 rounded-lg p-4 bg-white space-y-4">
              <div>
                <label
                  htmlFor="court-exact-speakers"
                  className="flex items-center gap-2 text-sm font-medium text-epam-gray-700 mb-1"
                >
                  <Users size={15} className="text-epam-red" />
                  {copy.courtExactSpeakers}
                </label>
                <input
                  id="court-exact-speakers"
                  type="number"
                  min={1}
                  max={40}
                  value={courtExactSpeakers}
                  onChange={(e) => setCourtExactSpeakers(e.target.value)}
                  placeholder="4"
                  className="w-32 px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red"
                />
                <p className="mt-1 text-xs text-epam-gray-400">
                  {copy.courtExactHelp}
                </p>
              </div>

              <div>
                <label
                  htmlFor="court-participants"
                  className="flex items-center gap-2 text-sm font-medium text-epam-gray-700 mb-1"
                >
                  <Users size={15} className="text-epam-gray-400" />
                  {copy.courtParticipants}
                </label>
                <textarea
                  id="court-participants"
                  value={courtParticipants}
                  onChange={(e) => setCourtParticipants(e.target.value)}
                  rows={3}
                  placeholder={copy.courtParticipantsPlaceholder}
                  className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red placeholder:text-epam-gray-300"
                />
              </div>

              <div>
                <label
                  htmlFor="court-dictionary"
                  className="flex items-center gap-2 text-sm font-medium text-epam-gray-700 mb-1"
                >
                  <BookOpen size={15} className="text-epam-gray-400" />
                  {copy.courtDictionary}
                </label>
                <textarea
                  id="court-dictionary"
                  value={courtDictionary}
                  onChange={(e) => setCourtDictionary(e.target.value)}
                  rows={4}
                  placeholder={copy.courtDictionaryPlaceholder}
                  className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red placeholder:text-epam-gray-300"
                />
                <p className="mt-1 text-xs text-epam-gray-400">
                  {copy.courtDictionaryHelp}
                </p>
              </div>
            </div>
          )}

          {/* Meeting context */}
          <div>
            <label
              htmlFor="meeting-context"
              className="block text-sm font-medium text-epam-gray-700 mb-1"
            >
              {copy.context}{" "}
              <span className="text-epam-gray-400 font-normal">({copy.optional})</span>
            </label>
            <textarea
              id="meeting-context"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              rows={3}
              placeholder={copy.contextPlaceholder}
              className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red placeholder:text-epam-gray-300"
            />
            <p className="mt-1 text-xs text-epam-gray-400">
              {copy.contextHelp}
            </p>
          </div>

          {/* Context file attachments */}
          <div>
            <label className="block text-sm font-medium text-epam-gray-700 mb-1">
              {copy.documents}{" "}
              <span className="text-epam-gray-400 font-normal">({copy.optional})</span>
            </label>

            {contextFiles.length > 0 && (
              <div className="mb-2 space-y-1">
                {contextFiles.map((cf, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 px-3 py-1.5 bg-epam-gray-50 rounded-lg text-sm"
                  >
                    <FileText size={14} className="text-epam-gray-400 shrink-0" />
                    <span className="text-epam-gray-700 truncate flex-1">
                      {cf.name}
                    </span>
                    <span className="text-xs text-epam-gray-400 shrink-0">
                      {formatFileSize(cf.size)}
                    </span>
                    <button
                      onClick={() => removeContextFile(idx)}
                      className="p-0.5 text-epam-gray-400 hover:text-red-500"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <input
              ref={contextFileInputRef}
              type="file"
              accept={ALLOWED_CONTEXT_EXT.join(",")}
              multiple
              onChange={handleContextFileAdd}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => contextFileInputRef.current?.click()}
              className="flex items-center gap-2 px-3 py-2 border border-dashed border-epam-gray-300 rounded-lg text-xs text-epam-gray-500 hover:border-epam-gray-400 hover:text-epam-gray-600 transition-colors"
            >
              <Paperclip size={14} />
              {copy.attachDocuments}
            </button>
            <p className="mt-1 text-xs text-epam-gray-400">
              {copy.documentsHelp}
            </p>
          </div>

          {/* Expected speakers — bucket dropdown.
              Tight ranges around expected count help pyannote VBx
              clustering converge correctly (per EPAM IT recommendation
              to use --min-speakers / --max-speakers with a narrow span). */}
          {meetingType !== "court_hearing" && (
          <div>
            <label
              htmlFor="speaker-bucket"
              className="block text-sm font-medium text-epam-gray-700 mb-1"
            >
              {copy.expectedSpeakers}
            </label>
            <select
              id="speaker-bucket"
              value={speakerBucket}
              onChange={(e) =>
                setSpeakerBucket(e.target.value as SpeakerBucket)
              }
              className="w-48 px-3 py-2 border border-epam-gray-300 rounded-lg text-sm text-epam-gray-700 bg-white focus:outline-none focus:ring-2 focus:ring-epam-red/30 focus:border-epam-red"
            >
              {SPEAKER_BUCKETS_IN_ORDER.map((b) => (
                <option key={b} value={b}>
                  {SPEAKER_BUCKET_COPY[language][b]}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-epam-gray-400">
              {copy.expectedSpeakersHelp}
            </p>
          </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle size={16} className="shrink-0" />
          {error}
        </div>
      )}

      {/* GPU/VRAM status — visible only when memory is tight or
          unavailable. Healthy state hidden so we don't clutter the
          page when there's nothing to worry about. */}
      <div className="mt-6">
        <GpuStatus refreshSec={5} hideWhenHealthy />
      </div>

      {/* Upload button */}
      <div className="mt-4 flex flex-col items-end gap-2">
        <button
          onClick={handleUpload}
          disabled={!file || uploading || vramLow}
          className="flex items-center gap-2 px-6 py-2.5 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title={
            vramLow
              ? copy.vramLowTitle
              : ""
          }
        >
          {uploading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              {copy.uploading}
            </>
          ) : (
            <>
              <CheckCircle size={16} />
              {copy.start}
            </>
          )}
        </button>
        {vramLow && (
          <p className="text-xs text-red-600">
            {copy.vramLow}
          </p>
        )}
      </div>

      {/* Security note */}
      <div className="mt-10 p-4 bg-epam-gray-100 rounded-lg">
        <p className="text-xs text-epam-gray-500 leading-relaxed">
          <strong className="text-epam-gray-600">{copy.securityLabel}</strong>{" "}
          {copy.security}
        </p>
      </div>
    </div>
  );
}
