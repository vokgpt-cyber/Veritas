/**
 * Admin audit log viewer (Sprint 2026-04-30, task #42).
 *
 * Read-only table over the per-user JSONL audit log. Loads the most
 * recent 200 entries from GET /api/system/audit (admin-only) with
 * optional filters: action name, username substring, since-timestamp.
 *
 * Useful for compliance audits in the law firm setting — every login,
 * every meeting upload/download/delete, every system event lands here.
 *
 * The table is intentionally simple: no client-side sorting, no row
 * actions. Reading is enough — write/edit on audit data is a
 * compliance violation we never want to enable.
 */

import { useEffect, useState } from "react";
import {
  Loader2,
  AlertCircle,
  RefreshCw,
  Filter,
  Download,
  ShieldCheck,
} from "lucide-react";
import { getToken } from "../api/client";

interface AuditEntry {
  timestamp: string;
  action: string;
  username: string;
  ip_address: string;
  resource_type?: string;
  resource_id?: string;
  details?: Record<string, unknown>;
  outcome?: string;
}

const PAGE_SIZE = 200;

const COMMON_ACTIONS = [
  "",
  "auth.login_success",
  "auth.login_failure",
  "auth.token_verify",
  "meeting.upload",
  "meeting.download",
  "meeting.delete",
  "speaker.enroll",
  "speaker.delete",
  "system.startup",
  "system.shutdown",
];

function formatTimestamp(iso: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("ru-RU", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function exportCsv(entries: AuditEntry[]): void {
  // Minimal CSV — enough for compliance reviewers to pull into Excel.
  // Quote every cell to handle commas / newlines in the details JSON.
  const header = [
    "timestamp",
    "action",
    "username",
    "ip_address",
    "resource_type",
    "resource_id",
    "outcome",
    "details",
  ];
  const escape = (v: unknown): string => {
    const s = v == null ? "" : typeof v === "string" ? v : JSON.stringify(v);
    return '"' + s.replace(/"/g, '""') + '"';
  };
  const rows = entries.map((e) =>
    [
      e.timestamp,
      e.action,
      e.username,
      e.ip_address,
      e.resource_type,
      e.resource_id,
      e.outcome,
      e.details,
    ]
      .map(escape)
      .join(","),
  );
  const blob = new Blob([header.join(",") + "\n" + rows.join("\n")], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `veritas_audit_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function AuditPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [actionFilter, setActionFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [sinceFilter, setSinceFilter] = useState("");

  async function load(opts: { offset?: number } = {}) {
    setLoading(true);
    setError(null);
    const newOffset = opts.offset ?? offset;
    try {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(newOffset),
      });
      if (actionFilter) params.append("action", actionFilter);
      if (userFilter) params.append("user_filter", userFilter);
      if (sinceFilter) params.append("since", sinceFilter);

      const token = getToken();
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const response = await fetch(`/api/system/audit?${params.toString()}`, {
        headers,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Failed: ${response.status}`);
      }
      const data: { total: number; entries: AuditEntry[] } =
        await response.json();
      setEntries(data.entries);
      setTotal(data.total);
      setOffset(newOffset);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Load failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load({ offset: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyFilters() {
    load({ offset: 0 });
  }

  function clearFilters() {
    setActionFilter("");
    setUserFilter("");
    setSinceFilter("");
    setTimeout(() => load({ offset: 0 }), 0);
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-heading text-2xl text-epam-gray-900 flex items-center gap-2">
            <ShieldCheck size={22} className="text-epam-red" />
            Журнал аудита
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            Только для администраторов. Записи о входах, операциях с
            совещаниями и системных событиях.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => exportCsv(entries)}
            disabled={entries.length === 0}
            className="flex items-center gap-2 px-4 py-2 border border-epam-gray-300 text-epam-gray-700 rounded-lg text-sm font-medium hover:bg-epam-gray-100 transition-colors disabled:opacity-50"
          >
            <Download size={14} />
            Экспорт CSV
          </button>
          <button
            onClick={() => load()}
            className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
            title="Обновить"
          >
            <RefreshCw size={18} />
          </button>
        </div>
      </div>

      {/* Filter bar */}
      <div className="bg-white border border-epam-gray-200 rounded-xl p-4 mb-6">
        <div className="flex items-center gap-2 mb-3">
          <Filter size={14} className="text-epam-gray-400" />
          <span className="text-sm font-medium text-epam-gray-700">
            Фильтры
          </span>
        </div>
        <div className="grid md:grid-cols-4 gap-3">
          <div>
            <label className="block text-xs text-epam-gray-500 mb-1">
              Действие
            </label>
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              className="w-full px-3 py-2 border border-epam-gray-300 rounded text-sm focus:outline-none focus:border-epam-red"
            >
              {COMMON_ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {a || "Все"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-epam-gray-500 mb-1">
              Пользователь содержит
            </label>
            <input
              type="text"
              value={userFilter}
              onChange={(e) => setUserFilter(e.target.value)}
              placeholder="например: ivanov"
              className="w-full px-3 py-2 border border-epam-gray-300 rounded text-sm focus:outline-none focus:border-epam-red"
            />
          </div>
          <div>
            <label className="block text-xs text-epam-gray-500 mb-1">
              С даты (ISO)
            </label>
            <input
              type="datetime-local"
              value={sinceFilter}
              onChange={(e) => setSinceFilter(e.target.value)}
              className="w-full px-3 py-2 border border-epam-gray-300 rounded text-sm focus:outline-none focus:border-epam-red"
            />
          </div>
          <div className="flex items-end gap-2">
            <button
              onClick={applyFilters}
              className="flex-1 px-4 py-2 bg-epam-red text-white rounded text-sm font-medium hover:bg-epam-red-dark"
            >
              Применить
            </button>
            <button
              onClick={clearFilters}
              className="px-4 py-2 border border-epam-gray-300 text-epam-gray-700 rounded text-sm font-medium hover:bg-epam-gray-100"
            >
              Сбросить
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <Loader2 size={24} className="animate-spin text-epam-red" />
        </div>
      ) : entries.length === 0 ? (
        <div className="text-center py-16 text-epam-gray-400 text-sm">
          Нет записей по выбранным фильтрам.
        </div>
      ) : (
        <>
          <p className="text-xs text-epam-gray-500 mb-3">
            Показано {entries.length} из {total} записей
            {offset > 0 && <> (с позиции {offset})</>}
          </p>
          <div className="bg-white border border-epam-gray-200 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-epam-gray-50">
                <tr>
                  <th className="text-left px-4 py-2 font-medium text-epam-gray-600">
                    Время
                  </th>
                  <th className="text-left px-4 py-2 font-medium text-epam-gray-600">
                    Действие
                  </th>
                  <th className="text-left px-4 py-2 font-medium text-epam-gray-600">
                    Пользователь
                  </th>
                  <th className="text-left px-4 py-2 font-medium text-epam-gray-600">
                    Ресурс
                  </th>
                  <th className="text-left px-4 py-2 font-medium text-epam-gray-600">
                    IP
                  </th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e, i) => (
                  <tr
                    key={i}
                    className="border-t border-epam-gray-100 hover:bg-epam-gray-50"
                  >
                    <td className="px-4 py-2 text-epam-gray-700 whitespace-nowrap font-mono text-xs">
                      {formatTimestamp(e.timestamp)}
                    </td>
                    <td className="px-4 py-2 text-epam-gray-700 font-mono text-xs">
                      {e.action}
                    </td>
                    <td className="px-4 py-2 text-epam-gray-700">
                      {e.username || <em className="text-epam-gray-400">—</em>}
                    </td>
                    <td className="px-4 py-2 text-epam-gray-500 text-xs">
                      {e.resource_type ? (
                        <>
                          {e.resource_type}
                          {e.resource_id && (
                            <span className="text-epam-gray-400">
                              {" "}
                              {e.resource_id}
                            </span>
                          )}
                        </>
                      ) : (
                        <em className="text-epam-gray-400">—</em>
                      )}
                    </td>
                    <td className="px-4 py-2 text-epam-gray-500 text-xs font-mono">
                      {e.ip_address || <em className="text-epam-gray-400">—</em>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="flex justify-between items-center mt-4 text-sm">
              <button
                onClick={() => load({ offset: Math.max(0, offset - PAGE_SIZE) })}
                disabled={offset <= 0}
                className="px-3 py-1.5 border border-epam-gray-300 text-epam-gray-700 rounded disabled:opacity-50"
              >
                ← Предыдущие
              </button>
              <span className="text-epam-gray-500 text-xs">
                Страница {Math.floor(offset / PAGE_SIZE) + 1} из{" "}
                {Math.ceil(total / PAGE_SIZE)}
              </span>
              <button
                onClick={() => load({ offset: offset + PAGE_SIZE })}
                disabled={offset + PAGE_SIZE >= total}
                className="px-3 py-1.5 border border-epam-gray-300 text-epam-gray-700 rounded disabled:opacity-50"
              >
                Следующие →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
