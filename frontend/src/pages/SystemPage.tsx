/**
 * System status dashboard showing health, GPU, and job statistics.
 */

import { useEffect, useState } from "react";
import {
  Activity,
  Cpu,
  HardDrive,
  Thermometer,
  Loader2,
  AlertCircle,
  RefreshCw,
  Server,
  BarChart3,
} from "lucide-react";
import { getSystemHealth, getGpuStatus, getSystemStats } from "../api/client";
import type { SystemHealth, GpuStatus, SystemStats } from "../types/api";

function ProgressBar({
  value,
  max,
  color = "bg-epam-red",
}: {
  value: number;
  max: number;
  color?: string;
}) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="h-2 bg-epam-gray-200 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${Math.min(pct, 100)}%` }}
      />
    </div>
  );
}

export default function SystemPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [gpu, setGpu] = useState<GpuStatus | null>(null);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    try {
      const [h, g, s] = await Promise.all([
        getSystemHealth().catch(() => null),
        getGpuStatus().catch(() => null),
        getSystemStats().catch(() => null),
      ]);
      setHealth(h);
      setGpu(g);
      setStats(s);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load system info");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const timer = setInterval(load, 10_000);
    return () => clearInterval(timer);
  }, []);

  if (loading && !health && !gpu && !stats) {
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
            System Status
          </h1>
          <p className="text-sm text-epam-gray-500 mt-1">
            On-premise hardware monitoring
          </p>
        </div>
        <button
          onClick={load}
          className="p-2 text-epam-gray-400 hover:text-epam-gray-600 transition-colors"
          title="Refresh"
        >
          <RefreshCw size={18} />
        </button>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 text-sm text-red-600">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* System Health */}
        {health && (
          <div className="bg-white border border-epam-gray-200 rounded-xl p-6">
            <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
              <Server size={18} className="text-epam-red" />
              System Health
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-epam-gray-600 flex items-center gap-1">
                    <Cpu size={14} /> CPU
                  </span>
                  <span className="font-medium">
                    {health.cpu_percent.toFixed(1)}%
                  </span>
                </div>
                <ProgressBar
                  value={health.cpu_percent}
                  max={100}
                  color={health.cpu_percent > 80 ? "bg-red-500" : "bg-epam-red"}
                />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-epam-gray-600 flex items-center gap-1">
                    <HardDrive size={14} /> RAM
                  </span>
                  <span className="font-medium">
                    {health.memory_percent.toFixed(1)}% &middot;{" "}
                    {health.memory_available_gb.toFixed(1)} GB free
                  </span>
                </div>
                <ProgressBar
                  value={health.memory_percent}
                  max={100}
                  color={health.memory_percent > 85 ? "bg-red-500" : "bg-blue-500"}
                />
              </div>

              <div className="pt-2 border-t border-epam-gray-100">
                <div className="flex justify-between text-sm">
                  <span className="text-epam-gray-600">Status</span>
                  <span className="text-green-600 font-medium">
                    {health.status}
                  </span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-epam-gray-600">Active Jobs</span>
                  <span className="font-medium">{health.active_jobs}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* GPU Status */}
        {gpu && (
          <div className="bg-white border border-epam-gray-200 rounded-xl p-6">
            <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
              <Activity size={18} className="text-epam-red" />
              GPU Status
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-epam-gray-600">VRAM</span>
                  <span className="font-medium">
                    {gpu.vram_used_gb.toFixed(1)} / {gpu.vram_total_gb.toFixed(1)} GB
                  </span>
                </div>
                <ProgressBar
                  value={gpu.vram_used_gb}
                  max={gpu.vram_total_gb}
                  color={
                    gpu.vram_used_gb / gpu.vram_total_gb > 0.85
                      ? "bg-red-500"
                      : "bg-green-500"
                  }
                />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-epam-gray-600">GPU Utilization</span>
                  <span className="font-medium">
                    {gpu.gpu_utilization.toFixed(0)}%
                  </span>
                </div>
                <ProgressBar value={gpu.gpu_utilization} max={100} color="bg-purple-500" />
              </div>

              <div className="pt-2 border-t border-epam-gray-100">
                <div className="flex justify-between text-sm">
                  <span className="text-epam-gray-600 flex items-center gap-1">
                    <Thermometer size={14} /> Temperature
                  </span>
                  <span
                    className={`font-medium ${gpu.temperature > 80 ? "text-red-600" : ""}`}
                  >
                    {gpu.temperature.toFixed(0)}°C
                  </span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-epam-gray-600">Device</span>
                  <span className="font-medium text-xs">
                    {gpu.device_name}
                  </span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-epam-gray-600">Telemetry</span>
                  <span className="font-medium text-xs">
                    {gpu.source ?? "server"}
                  </span>
                </div>
                {gpu.current_model && (
                  <div className="flex justify-between text-sm mt-1">
                    <span className="text-epam-gray-600">Loaded Model</span>
                    <span className="font-medium text-xs">
                      {gpu.current_model}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Job Statistics */}
        {stats && (
          <div className="bg-white border border-epam-gray-200 rounded-xl p-6 md:col-span-2">
            <h2 className="font-heading text-lg text-epam-gray-900 mb-4 flex items-center gap-2">
              <BarChart3 size={18} className="text-epam-red" />
              Job Statistics
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center p-3 bg-epam-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-epam-gray-900">
                  {stats.total_jobs}
                </p>
                <p className="text-xs text-epam-gray-500 mt-1">Total</p>
              </div>
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <p className="text-2xl font-bold text-green-700">
                  {stats.completed_jobs}
                </p>
                <p className="text-xs text-green-600 mt-1">Completed</p>
              </div>
              <div className="text-center p-3 bg-amber-50 rounded-lg">
                <p className="text-2xl font-bold text-amber-700">
                  {stats.processing_jobs}
                </p>
                <p className="text-xs text-amber-600 mt-1">Processing</p>
              </div>
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <p className="text-2xl font-bold text-blue-700">
                  {stats.pending_jobs}
                </p>
                <p className="text-xs text-blue-600 mt-1">Pending</p>
              </div>
              <div className="text-center p-3 bg-red-50 rounded-lg">
                <p className="text-2xl font-bold text-red-700">
                  {stats.failed_jobs}
                </p>
                <p className="text-xs text-red-600 mt-1">Failed</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {!health && !gpu && !stats && !error && (
        <div className="text-center py-20 text-epam-gray-400 text-sm">
          Unable to connect to the backend. Ensure VERITAS is running.
        </div>
      )}
    </div>
  );
}
