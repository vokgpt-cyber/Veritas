/**
 * WebSocket hook for real-time meeting progress updates.
 *
 * Connects to ws://host/ws/meetings/{jobId} and provides
 * live progress, state, and ETA updates.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import type { ProgressMessage, PipelineState } from "../types/api";

interface WebSocketState {
  progress: number;
  state: PipelineState;
  currentStage: string;
  etaSeconds: number | null;
  connected: boolean;
  error: string | null;
}

const PING_INTERVAL = 5_000;
const RECONNECT_DELAY = 3_000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useWebSocket(jobId: string | null): WebSocketState {
  const [wsState, setWsState] = useState<WebSocketState>({
    progress: 0,
    state: "uploaded",
    currentStage: "",
    etaSeconds: null,
    connected: false,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!jobId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/meetings/${jobId}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectCountRef.current = 0;
        setWsState((prev) => ({ ...prev, connected: true, error: null }));

        // Start ping + status polling interval
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "status" }));
          }
        }, PING_INTERVAL);

        // Request initial status
        ws.send(JSON.stringify({ type: "status" }));
      };

      ws.onmessage = (event) => {
        try {
          const msg: ProgressMessage = JSON.parse(event.data);

          if (msg.type === "pong") return;

          if (msg.type === "progress" || msg.type === "status") {
            setWsState((prev) => ({
              ...prev,
              progress: msg.progress,
              state: msg.state,
              currentStage: msg.current_stage,
              etaSeconds: msg.eta_seconds,
            }));
          }

          if (msg.type === "error") {
            setWsState((prev) => ({
              ...prev,
              state: "error",
              error: msg.current_stage || "Processing error",
            }));
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        setWsState((prev) => ({ ...prev, connected: false }));

        if (pingRef.current) {
          clearInterval(pingRef.current);
          pingRef.current = null;
        }

        // Reconnect if not completed/errored
        if (
          reconnectCountRef.current < MAX_RECONNECT_ATTEMPTS &&
          wsState.state !== "completed" &&
          wsState.state !== "error"
        ) {
          reconnectCountRef.current += 1;
          reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY);
        }
      };

      ws.onerror = () => {
        setWsState((prev) => ({
          ...prev,
          error: "WebSocket connection error",
        }));
      };
    } catch {
      setWsState((prev) => ({
        ...prev,
        error: "Failed to create WebSocket connection",
      }));
    }
  }, [jobId, wsState.state]);

  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pingRef.current) {
        clearInterval(pingRef.current);
        pingRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  return wsState;
}
