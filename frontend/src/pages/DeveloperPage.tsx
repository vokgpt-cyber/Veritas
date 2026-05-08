import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Download,
  Loader2,
  Save,
  ShieldAlert,
  SlidersHorizontal,
} from "lucide-react";
import {
  getDeveloperSettings,
  getDeveloperTechnicalLogUrl,
  getToken,
  updateDeveloperActive,
  updateDeveloperPrompt,
} from "../api/client";
import { useAuth } from "../context/AuthContext";
import type {
  DeveloperModelCategory,
  DeveloperPrompt,
  DeveloperSettings,
} from "../types/api";

const categoryLabels: Record<DeveloperModelCategory, string> = {
  asr: "Транскрибация (ASR)",
  diarization: "Диаризация",
  llm: "Саммаризация / протокол",
};

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export default function DeveloperPage() {
  const { role, username, authRequired } = useAuth();
  const canUseDeveloper =
    role === "admin" || role === "developer" || username === "admin" || !authRequired;

  const [settings, setSettings] = useState<DeveloperSettings | null>(null);
  const [activeDraft, setActiveDraft] =
    useState<DeveloperSettings["active"] | null>(null);
  const [selectedPromptId, setSelectedPromptId] = useState("");
  const [promptDraft, setPromptDraft] = useState<DeveloperPrompt | null>(null);
  const [jobId, setJobId] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!canUseDeveloper) {
      setLoading(false);
      return;
    }
    getDeveloperSettings()
      .then((data) => {
        setSettings(data);
        setActiveDraft(data.active);
        const firstPrompt = data.prompts[0];
        if (firstPrompt) {
          setSelectedPromptId(firstPrompt.id);
          setPromptDraft(firstPrompt);
        }
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Developer settings failed"),
      )
      .finally(() => setLoading(false));
  }, [canUseDeveloper]);

  useEffect(() => {
    if (!settings || !selectedPromptId) return;
    const prompt = settings.prompts.find((item) => item.id === selectedPromptId);
    if (prompt) setPromptDraft(prompt);
  }, [selectedPromptId, settings]);

  const selectedModels = useMemo(() => {
    if (!settings || !activeDraft) return {};
    return {
      asr: settings.catalog.asr.find((item) => item.id === activeDraft.asr),
      diarization: settings.catalog.diarization.find(
        (item) => item.id === activeDraft.diarization,
      ),
      llm: settings.catalog.llm.find((item) => item.id === activeDraft.llm),
    };
  }, [activeDraft, settings]);

  async function saveActiveSettings() {
    if (!activeDraft) return;
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const updated = await updateDeveloperActive(activeDraft);
      setSettings(updated);
      setActiveDraft(updated.active);
      setMessage("Настройки моделей сохранены. Следующая обработка возьмет новый набор.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setSaving(false);
    }
  }

  async function savePrompt() {
    if (!promptDraft) return;
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const updated = await updateDeveloperPrompt(promptDraft.id, {
        enabled: promptDraft.enabled,
        system_template: promptDraft.system_template,
        user_template: promptDraft.user_template,
        notes: promptDraft.notes,
      });
      setSettings((current) =>
        current
          ? {
              ...current,
              prompts: current.prompts.map((item) =>
                item.id === updated.id ? updated : item,
              ),
            }
          : current,
      );
      setPromptDraft(updated);
      setMessage("Промпт сохранен.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save prompt");
    } finally {
      setSaving(false);
    }
  }

  async function exportTechnicalLog() {
    const cleanJobId = jobId.trim();
    if (!cleanJobId) return;
    setError(null);
    setMessage(null);
    try {
      const token = getToken();
      const response = await fetch(getDeveloperTechnicalLogUrl(cleanJobId), {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Export failed: ${response.status}`);
      }
      downloadBlob(await response.blob(), `veritas_technical_log_${cleanJobId}.zip`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Technical log export failed");
    }
  }

  if (!canUseDeveloper) {
    return (
      <div className="p-6 max-w-3xl">
        <div className="flex items-center gap-3 rounded-lg border border-epam-gray-200 bg-white p-4 text-epam-gray-700">
          <ShieldAlert size={20} className="text-epam-red" />
          Developer access required.
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 size={24} className="animate-spin text-epam-red" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div>
        <h1 className="font-heading text-2xl text-epam-gray-900">
          Настройки разработчика
        </h1>
        <p className="mt-1 text-sm text-epam-gray-500">
          Закрытое меню для IT: выбор проверенных моделей, правка промптов и
          выгрузка технических логов без содержания документов.
        </p>
      </div>

      {(message || error) && (
        <div
          className={`flex items-center gap-2 rounded-md border px-4 py-3 text-sm ${
            error
              ? "border-red-200 bg-red-50 text-red-700"
              : "border-green-200 bg-green-50 text-green-700"
          }`}
        >
          <AlertCircle size={16} />
          {error || message}
        </div>
      )}

      {settings && activeDraft && (
        <section className="rounded-lg border border-epam-gray-200 bg-white">
          <div className="border-b border-epam-gray-200 px-4 py-3">
            <h2 className="flex items-center gap-2 text-base font-semibold text-epam-gray-900">
              <SlidersHorizontal size={17} className="text-epam-red" />
              Профиль и модели
            </h2>
          </div>
          <div className="space-y-4 p-4">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                Профиль оборудования
              </span>
              <select
                value={activeDraft.profile}
                onChange={(event) =>
                  setActiveDraft((current) =>
                    current ? { ...current, profile: event.target.value } : current,
                  )
                }
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
              >
                {settings.profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>

            {(["asr", "diarization", "llm"] as DeveloperModelCategory[]).map(
              (category) => (
                <label key={category} className="block">
                  <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                    {categoryLabels[category]}
                  </span>
                  <select
                    value={activeDraft[category]}
                    onChange={(event) =>
                      setActiveDraft((current) =>
                        current
                          ? { ...current, [category]: event.target.value }
                          : current,
                      )
                    }
                    className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
                  >
                    {settings.catalog[category].map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} · {model.status}
                      </option>
                    ))}
                  </select>
                  {selectedModels[category]?.notes && (
                    <p className="mt-1 text-xs text-epam-gray-500">
                      {selectedModels[category]?.notes}
                    </p>
                  )}
                </label>
              ),
            )}

            <button
              onClick={saveActiveSettings}
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-md bg-epam-red px-4 py-2 text-sm font-semibold text-white hover:bg-epam-red-dark disabled:opacity-50"
            >
              {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
              Сохранить модели
            </button>
          </div>
        </section>
      )}

      {settings && promptDraft && (
        <section className="rounded-lg border border-epam-gray-200 bg-white">
          <div className="border-b border-epam-gray-200 px-4 py-3">
            <h2 className="text-base font-semibold text-epam-gray-900">
              Промпты
            </h2>
          </div>
          <div className="space-y-4 p-4">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                Промпт
              </span>
              <select
                value={selectedPromptId}
                onChange={(event) => setSelectedPromptId(event.target.value)}
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
              >
                {settings.prompts.map((prompt) => (
                  <option key={prompt.id} value={prompt.id}>
                    {prompt.title} · v{prompt.version}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex items-center gap-2 text-sm text-epam-gray-700">
              <input
                type="checkbox"
                checked={promptDraft.enabled}
                onChange={(event) =>
                  setPromptDraft((current) =>
                    current ? { ...current, enabled: event.target.checked } : current,
                  )
                }
                className="accent-epam-red"
              />
              Использовать этот промпт вместо встроенного
            </label>

            <p className="text-xs text-epam-gray-500">
              Доступные переменные: {promptDraft.placeholders.join(", ")}.
              Если промпт выключен, VERITAS использует встроенный production-промпт.
            </p>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                System prompt
              </span>
              <textarea
                value={promptDraft.system_template}
                onChange={(event) =>
                  setPromptDraft((current) =>
                    current
                      ? { ...current, system_template: event.target.value }
                      : current,
                  )
                }
                rows={5}
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 font-mono text-xs text-epam-gray-900 focus:border-epam-red focus:outline-none"
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                User prompt
              </span>
              <textarea
                value={promptDraft.user_template}
                onChange={(event) =>
                  setPromptDraft((current) =>
                    current
                      ? { ...current, user_template: event.target.value }
                      : current,
                  )
                }
                rows={10}
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 font-mono text-xs text-epam-gray-900 focus:border-epam-red focus:outline-none"
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-epam-gray-500">
                Notes
              </span>
              <input
                value={promptDraft.notes}
                onChange={(event) =>
                  setPromptDraft((current) =>
                    current ? { ...current, notes: event.target.value } : current,
                  )
                }
                className="w-full rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
              />
            </label>

            <button
              onClick={savePrompt}
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-md bg-epam-red px-4 py-2 text-sm font-semibold text-white hover:bg-epam-red-dark disabled:opacity-50"
            >
              {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
              Сохранить промпт
            </button>
          </div>
        </section>
      )}

      <section className="rounded-lg border border-epam-gray-200 bg-white">
        <div className="border-b border-epam-gray-200 px-4 py-3">
          <h2 className="text-base font-semibold text-epam-gray-900">
            Технический лог сессии
          </h2>
        </div>
        <div className="space-y-3 p-4">
          <p className="text-sm text-epam-gray-500">
            Архив содержит настройки пайплайна и технические параметры запуска,
            но не содержит текст стенограммы, приложений, словарей или протокола.
          </p>
          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              value={jobId}
              onChange={(event) => setJobId(event.target.value)}
              placeholder="Job ID"
              className="flex-1 rounded-md border border-epam-gray-300 bg-white px-3 py-2 text-sm text-epam-gray-900 focus:border-epam-red focus:outline-none"
            />
            <button
              onClick={exportTechnicalLog}
              disabled={!jobId.trim()}
              className="inline-flex items-center justify-center gap-2 rounded-md border border-epam-gray-300 px-4 py-2 text-sm font-semibold text-epam-gray-700 hover:bg-epam-gray-100 disabled:opacity-50"
            >
              <Download size={16} />
              Скачать лог
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
