/**
 * Live GPU memory status indicator.
 *
 * Polls /api/system/gpu every refreshSec seconds, shows free/used/total
 * VRAM with a colour-coded bar:
 *   - green when free >= 16 GB
 *   - amber when 14-16 GB (quality ASR may run, but with little margin)
 *   - red when < 14 GB (quality pipeline should not start)
 *
 * Used on UploadPage (gates the submit button), ProcessingPage (lets
 * the user see why VRAM ran out mid-run), and SystemPage (general
 * health). When the GPU isn't reachable (CPU-only or driver issue),
 * collapses to a small grey "GPU недоступен" line — never blocks UI.
 */
import { useEffect, useState } from "react";
import { Cpu, AlertTriangle, RefreshCw } from "lucide-react";
import { getGpuStatus } from "../api/client";
import type { GpuStatus as GpuStatusType } from "../types/api";

interface Props {
  /** Polling interval in seconds. 0 disables auto-refresh. */
  refreshSec?: number;
  /** Hide when GPU is healthy (free >= 16 GB). Useful on UploadPage
   * where the indicator only matters when there's a problem. */
  hideWhenHealthy?: boolean;
  /** Optional className passthrough. */
  className?: string;
}

export default function GpuStatus({
  refreshSec = 5,
  hideWhenHealthy = false,
  className = "",
}: Props) {
  const [status, setStatus] = useState<GpuStatusType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  async function refresh() {
    setRefreshing(true);
    try {
      const s = await getGpuStatus();
      setStatus(s);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "GPU недоступен");
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    refresh();
    if (refreshSec > 0) {
      const id = setInterval(refresh, refreshSec * 1000);
      return () => clearInterval(id);
    }
    return undefined;
  }, [refreshSec]);

  if (error || !status || !status.is_available) {
    if (hideWhenHealthy) return null;
    return (
      <div
        className={`text-xs text-epam-gray-400 flex items-center gap-2 ${className}`}
      >
        <Cpu size={12} />
        GPU недоступен{error ? ` (${error})` : ""}
      </div>
    );
  }

  const free = status.vram_free_gb;
  const total = status.vram_total_gb;
  const used = status.vram_used_gb;
  const freeFraction = total > 0 ? free / total : 0;

  // Color thresholds:
  //   <14 GB free  -> red   (quality GigaAM batch needs more headroom)
  //   14-16 GB free -> amber (can run, but close other GPU apps if possible)
  //   >=16 GB free -> green
  let tone: "green" | "amber" | "red";
  if (free < 14) tone = "red";
  else if (free < 16) tone = "amber";
  else tone = "green";

  if (hideWhenHealthy && tone === "green") return null;

  const colorClasses: Record<typeof tone, { bar: string; text: string; bg: string; ring: string }> = {
    green: {
      bar: "bg-emerald-500",
      text: "text-emerald-700",
      bg: "bg-emerald-50",
      ring: "ring-emerald-200",
    },
    amber: {
      bar: "bg-amber-500",
      text: "text-amber-700",
      bg: "bg-amber-50",
      ring: "ring-amber-300",
    },
    red: {
      bar: "bg-red-500",
      text: "text-red-700",
      bg: "bg-red-50",
      ring: "ring-red-300",
    },
  };
  const c = colorClasses[tone];

  return (
    <div
      className={`rounded-lg border ${c.bg} ${c.ring} ring-1 px-3 py-2 ${className}`}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <div className={`flex items-center gap-2 text-xs font-medium ${c.text}`}>
          {tone === "red" ? (
            <AlertTriangle size={13} />
          ) : (
            <Cpu size={13} />
          )}
          Видеопамять GPU
        </div>
        <button
          onClick={refresh}
          disabled={refreshing}
          className={`text-xs ${c.text} hover:underline flex items-center gap-1 disabled:opacity-50`}
          title="Обновить"
        >
          <RefreshCw
            size={11}
            className={refreshing ? "animate-spin" : ""}
          />
          Обновить
        </button>
      </div>

      <div className="w-full h-1.5 bg-epam-gray-200 rounded overflow-hidden mb-1.5">
        <div
          className={`h-full ${c.bar} transition-all`}
          style={{ width: `${(1 - freeFraction) * 100}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-xs text-epam-gray-600 font-mono">
        <span>
          Свободно:{" "}
          <span className={`font-semibold ${c.text}`}>
            {free.toFixed(1)} ГБ
          </span>
        </span>
        <span className="text-epam-gray-400">
          {used.toFixed(1)} / {total.toFixed(1)} ГБ занято
        </span>
      </div>

      {tone === "red" && (
        <p className="mt-2 text-xs text-red-700 leading-relaxed">
          Меньше 14 ГБ свободно — качественный пайплайн не запустится. Закройте другие
          GPU-приложения (LM Studio, Stable Diffusion, ChatGPT desktop,
          игры в фоне) и нажмите «Обновить».
        </p>
      )}
      {tone === "amber" && (
        <p className="mt-2 text-xs text-amber-700 leading-relaxed">
          Памяти впритык. ASR может запуститься, но запас маленький.
          Желательно освободить ещё 2–3 ГБ перед запуском.
        </p>
      )}
    </div>
  );
}
