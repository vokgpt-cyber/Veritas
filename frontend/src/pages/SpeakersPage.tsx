/**
 * Voice profile management (Block 7, Sprint 2026-04-30).
 *
 * Two halves:
 *   1. Enrollment panel: record/import a 30-60s voice sample, fill
 *      name + department, submit.
 *   2. Profile list: edit metadata, add additional samples, delete.
 *
 * Recording uses the browser's MediaRecorder API. We default to
 * audio/webm output because every modern browser supports it and
 * the backend's pyannote/torchaudio chain decodes it via librosa
 * fallback. Samples are uploaded as Blob to POST /api/speakers/
 * which extracts the embedding, encrypts it, and returns the
 * profile metadata.
 *
 * Layout follows the rest of VERITAS — Tailwind, EPAM red accents,
 * Russian-language UI for the law firm.
 */

import { useEffect, useRef, useState } from "react";
import {
  Users,
  Trash2,
  Loader2,
  AlertCircle,
  RefreshCw,
  Mic,
  Square,
  Upload,
  CheckCircle,
  X,
  Edit2,
  Plus,
} from "lucide-react";
import {
  listSpeakers,
  deleteSpeaker,
  enrolSpeaker,
  addSpeakerSample,
  updateSpeakerProfile,
} from "../api/client";
import type { SpeakerProfile } from "../types/api";

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function formatTimer(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Minimum and recommended sample length. The backend pyannote/embedding
// model wants ≥ 8s of speech to converge; ≥ 30s is comfortable.
const MIN_SAMPLE_S = 8;
const RECOMMENDED_SAMPLE_S = 30;

export default function SpeakersPage() {
  // Profile list state
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  // Enrollment form state
  const [name, setName] = useState("");
  const [department, setDepartment] = useState("");
  const [position, setPosition] = useState("");
  const [isAdminDefault, setIsAdminDefault] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [recordedDuration, setRecordedDuration] = useState(0);

  // Recording state
  const [recording, setRecording] = useState(false);
  const [recTimer, setRecTimer] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  // Submission state
  const [submitting, setSubmitting] = useState(false);

  // Edit dialog state
  const [editing, setEditing] = useState<SpeakerProfile | null>(null);

  async function load() {
    setLoading(true);
    try {
      const data = await listSpeakers();
      setSpeakers(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load profiles");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    return () => {
      // Clean up any open mic stream on unmount
      stopStream();
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }

  async function handleStartRecording() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        "Браузер не поддерживает запись с микрофона. Загрузите аудиофайл вместо записи.",
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        setRecordedBlob(blob);
        if (recordedUrl) URL.revokeObjectURL(recordedUrl);
        setRecordedUrl(URL.createObjectURL(blob));
        setRecordedDuration(recTimer);
        stopStream();
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecTimer(0);
      setRecording(true);
      timerRef.current = window.setInterval(() => {
        setRecTimer((t) => t + 1);
      }, 1000);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Запись недоступна: ${err.message}`
          : "Не удалось получить доступ к микрофону",
      );
      stopStream();
    }
  }

  function handleStopRecording() {
    if (recorderRef.current && recording) {
      recorderRef.current.stop();
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRecording(false);
  }

  function handleClearSample() {
    if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    setRecordedBlob(null);
    setRecordedUrl(null);
    setRecordedDuration(0);
    setRecTimer(0);
  }

  async function handleFileImport(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setError(null);
    if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    setRecordedBlob(f);
    setRecordedUrl(URL.createObjectURL(f));
    // We don't know exact duration without decoding. Show 0 — backend
    // will reject if too short.
    setRecordedDuration(0);
    e.target.value = "";
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!name.trim()) {
      setError("Введите имя сотрудника");
      return;
    }
    setSubmitting(true);
    try {
      // If the source was a recording, normalise the filename suffix
      // so the backend's extension whitelist accepts it.
      const filename =
        recordedBlob instanceof File
          ? recordedBlob.name
          : "voice_sample.webm";
      await enrolSpeaker({
        name: name.trim(),
        department: department.trim(),
        position: position.trim(),
        isAdminDefault,
        audio: recordedBlob ?? undefined,
        audioFilename: recordedBlob ? filename : undefined,
      });
      setName("");
      setDepartment("");
      setPosition("");
      setIsAdminDefault(false);
      handleClearSample();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Регистрация не удалась");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: string, profileName: string) {
    if (
      !confirm(
        `Удалить голосовой профиль "${profileName}"? Это действие необратимо.`,
      )
    )
      return;
    setBusyId(id);
    try {
      await deleteSpeaker(id);
      setSpeakers((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Удаление не удалось");
    } finally {
      setBusyId(null);
    }
  }

  async function handleAddSample(speakerId: string, file: File) {
    setBusyId(speakerId);
    try {
      await addSpeakerSample(speakerId, file, file.name);
      await load();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Не удалось добавить голосовой образец",
      );
    } finally {
      setBusyId(null);
    }
  }

  async function handleSaveEdit(profile: SpeakerProfile) {
    try {
      await updateSpeakerProfile(profile.id, {
        name: profile.name,
        department: profile.department,
        position: profile.position || "",
        is_admin_default: profile.is_admin_default,
      });
      await load();
      setEditing(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось сохранить");
    }
  }

  const sampleTooShort = recordedDuration > 0 && recordedDuration < MIN_SAMPLE_S;

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-heading text-2xl text-epam-gray-900">
            Сотрудники и голоса
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            {speakers.length}{" "}
            {speakers.length === 1
              ? "зарегистрирован"
              : speakers.length >= 2 && speakers.length <= 4
                ? "зарегистрировано"
                : "зарегистрировано"}{" "}
            профил
            {speakers.length === 1 ? "ь" : speakers.length >= 5 ? "ей" : "я"}
          </p>
        </div>
        <button
          onClick={load}
          className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
          title="Обновить"
        >
          <RefreshCw size={18} />
        </button>
      </div>

      {error && (
        <div className="mb-4 flex items-start gap-2 text-sm text-red-600 p-3 bg-red-50 border border-red-200 rounded-lg">
          <AlertCircle size={16} className="shrink-0 mt-0.5" />
          <span className="flex-1">{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-600"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Enrollment panel */}
      <div className="bg-white border border-epam-gray-200 rounded-xl p-6 mb-8">
        <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
          <Plus size={18} className="text-epam-red" />
          Добавить сотрудника
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid md:grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-epam-gray-500 mb-1">
                Имя <span className="text-epam-red">*</span>
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Иванов А.В."
                className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:border-epam-red"
                disabled={submitting}
              />
            </div>
            <div>
              <label className="block text-xs text-epam-gray-500 mb-1">
                Подразделение
              </label>
              <input
                type="text"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                placeholder="Судебная практика"
                className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:border-epam-red"
                disabled={submitting}
              />
            </div>
            <div>
              <label className="block text-xs text-epam-gray-500 mb-1">
                Должность
              </label>
              <input
                type="text"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
                placeholder="Старший юрист"
                className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:border-epam-red"
                disabled={submitting}
              />
            </div>
          </div>

          <label className="flex items-start gap-3 rounded-lg border border-epam-gray-200 bg-epam-gray-50 px-3 py-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={isAdminDefault}
              onChange={(e) => setIsAdminDefault(e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-epam-red"
              disabled={submitting}
            />
            <span>
              <span className="block font-medium text-epam-gray-800">
                Обычно участвует в административных совещаниях
              </span>
              <span className="block text-xs text-epam-gray-500 mt-0.5">
                Такие сотрудники будут заранее отмечены при загрузке записи
                административного совещания.
              </span>
            </span>
          </label>

          {/* Recording controls */}
          <div className="border border-epam-gray-200 rounded-lg p-4">
            <p className="text-xs text-epam-gray-500 mb-3">
              Запишите или загрузите голосовой образец длительностью{" "}
              {RECOMMENDED_SAMPLE_S} секунд и более. Минимум —{" "}
              {MIN_SAMPLE_S} секунд.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              {!recording && !recordedBlob && (
                <>
                  <button
                    type="button"
                    onClick={handleStartRecording}
                    className="flex items-center gap-2 px-4 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors"
                  >
                    <Mic size={16} />
                    Записать
                  </button>
                  <span className="text-xs text-epam-gray-400">или</span>
                  <label className="flex items-center gap-2 px-4 py-2 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors cursor-pointer">
                    <Upload size={16} />
                    Загрузить файл
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={handleFileImport}
                      className="hidden"
                    />
                  </label>
                </>
              )}

              {recording && (
                <>
                  <button
                    type="button"
                    onClick={handleStopRecording}
                    className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition-colors"
                  >
                    <Square size={14} />
                    Остановить
                  </button>
                  <span className="flex items-center gap-2 text-sm text-red-600 font-mono">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    Запись {formatTimer(recTimer)}
                  </span>
                </>
              )}

              {recordedBlob && !recording && (
                <>
                  <CheckCircle size={18} className="text-emerald-600" />
                  <span className="text-sm text-epam-gray-700">
                    Образец готов
                    {recordedDuration > 0 &&
                      ` (${formatTimer(recordedDuration)})`}
                  </span>
                  {recordedUrl && (
                    <audio
                      src={recordedUrl}
                      controls
                      className="h-8 max-w-xs"
                    />
                  )}
                  <button
                    type="button"
                    onClick={handleClearSample}
                    className="text-xs text-epam-gray-500 hover:text-red-600 underline"
                  >
                    очистить
                  </button>
                </>
              )}
            </div>

            {sampleTooShort && (
              <p className="mt-2 text-xs text-amber-700">
                Образец слишком короткий ({formatTimer(recordedDuration)}).
                Нужно не менее {MIN_SAMPLE_S} секунд для надёжного отпечатка
                голоса.
              </p>
            )}
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={sampleTooShort || submitting || !name.trim()}
              className="flex items-center gap-2 px-5 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <CheckCircle size={16} />
              )}
              {submitting ? "Регистрация..." : "Зарегистрировать"}
            </button>
          </div>
        </form>
      </div>

      {/* Profile list */}
      {loading ? (
        <div className="flex items-center justify-center h-48">
          <Loader2 size={24} className="animate-spin text-epam-red" />
        </div>
      ) : speakers.length === 0 ? (
        <div className="text-center py-16">
          <Users size={48} className="mx-auto text-epam-gray-300 mb-4" />
          <p className="text-epam-gray-500 mb-2">
            Нет зарегистрированных голосов
          </p>
          <p className="text-xs text-epam-gray-400 max-w-sm mx-auto">
            После регистрации система будет автоматически распознавать
            этих сотрудников в записях совещаний и подставлять их имена
            вместо «SPEAKER_1», «SPEAKER_2» и т.д.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {speakers.map((speaker) => (
            <div
              key={speaker.id}
              className="flex items-center gap-4 p-4 bg-white border border-epam-gray-200 rounded-xl"
            >
              <div className="w-10 h-10 rounded-full bg-epam-red/10 flex items-center justify-center shrink-0">
                <span className="text-sm font-bold text-epam-red">
                  {speaker.name.charAt(0).toUpperCase()}
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-epam-gray-900 flex items-center gap-2">
                  {speaker.name}
                  {speaker.is_admin_default && (
                    <span className="text-[10px] uppercase tracking-wide text-epam-red border border-epam-red/30 rounded-full px-2 py-0.5">
                      Админсовещание
                    </span>
                  )}
                </p>
                <p className="text-xs text-epam-gray-400 mt-0.5 flex flex-wrap gap-x-3">
                  {speaker.department && <span>{speaker.department}</span>}
                  {speaker.position && <span>{speaker.position}</span>}
                  <span>Образцов: {speaker.sample_count}</span>
                  {speaker.match_count > 0 && (
                    <span>Совпадений: {speaker.match_count}</span>
                  )}
                  <span>Зарегистр.: {formatDate(speaker.registered_at)}</span>
                  {speaker.last_used_at && (
                    <span>Последний раз: {formatDate(speaker.last_used_at)}</span>
                  )}
                </p>
              </div>

              <label className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors cursor-pointer" title="Добавить ещё образец">
                <Plus size={16} />
                <input
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleAddSample(speaker.id, f);
                    e.target.value = "";
                  }}
                />
              </label>

              <button
                onClick={() => setEditing(speaker)}
                className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
                title="Редактировать"
              >
                <Edit2 size={16} />
              </button>

              <button
                onClick={() => handleDelete(speaker.id, speaker.name)}
                disabled={busyId === speaker.id}
                className="p-2 text-epam-gray-400 hover:text-red-500 transition-colors disabled:opacity-50"
                title="Удалить"
              >
                {busyId === speaker.id ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Trash2 size={16} />
                )}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Edit dialog */}
      {editing && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 max-w-md w-full">
            <h3 className="font-heading text-lg text-epam-gray-900 mb-4">
              Редактировать профиль
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-epam-gray-500 mb-1">
                  Имя
                </label>
                <input
                  type="text"
                  value={editing.name}
                  onChange={(e) =>
                    setEditing({ ...editing, name: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:border-epam-red"
                />
              </div>
              <div>
                <label className="block text-xs text-epam-gray-500 mb-1">
                  Подразделение
                </label>
                <input
                  type="text"
                  value={editing.department || ""}
                  onChange={(e) =>
                    setEditing({ ...editing, department: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:border-epam-red"
                />
              </div>
              <div>
                <label className="block text-xs text-epam-gray-500 mb-1">
                  Должность
                </label>
                <input
                  type="text"
                  value={editing.position || ""}
                  onChange={(e) =>
                    setEditing({ ...editing, position: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-epam-gray-300 rounded-lg text-sm focus:outline-none focus:border-epam-red"
                />
              </div>
              <label className="flex items-start gap-3 rounded-lg border border-epam-gray-200 bg-epam-gray-50 px-3 py-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={editing.is_admin_default}
                  onChange={(e) =>
                    setEditing({
                      ...editing,
                      is_admin_default: e.target.checked,
                    })
                  }
                  className="mt-0.5 h-4 w-4 accent-epam-red"
                />
                <span>
                  <span className="block font-medium text-epam-gray-800">
                    Обычно участвует в административных совещаниях
                  </span>
                  <span className="block text-xs text-epam-gray-500 mt-0.5">
                    Будет заранее выбран на экране загрузки админсовещания.
                  </span>
                </span>
              </label>
            </div>
            <div className="flex justify-end gap-2 mt-5">
              <button
                onClick={() => setEditing(null)}
                className="px-4 py-2 text-sm text-epam-gray-600 hover:text-epam-gray-900"
              >
                Отмена
              </button>
              <button
                onClick={() => handleSaveEdit(editing)}
                className="px-4 py-2 bg-epam-red text-white rounded-lg text-sm font-medium hover:bg-epam-red-dark"
              >
                Сохранить
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
