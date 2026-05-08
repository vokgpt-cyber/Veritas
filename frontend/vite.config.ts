import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Backend port. Default 8765 (VERITAS' chosen port — picked to avoid
// collisions with FastAPI / Django defaults on 8000 and assorted dev
// servers on 5000/8080). Override by setting EPAM_BACKEND_PORT in
// the environment (the launcher reads it and forwards).
const BACKEND_HOST = process.env.EPAM_BACKEND_HOST || "localhost";
const BACKEND_PORT = Number(process.env.EPAM_BACKEND_PORT) || 8765;

// Frontend port. Default 3000. Override via EPAM_FRONTEND_PORT —
// the launcher (START_VERITAS_MVP.bat) finds the next free port in
// the 3000..3010 range when 3000 is taken (e.g. when the user runs
// multiple projects in parallel) and passes it through here.
// strictPort=true so vite fails loudly instead of silently picking
// 3001 — that broke the launcher's "open browser on 3000" step
// (user reported 2026-05-04). When the launcher passes a free port
// it's already verified, so failing here means env mismatch.
const FRONTEND_PORT = Number(process.env.EPAM_FRONTEND_PORT) || 3000;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: FRONTEND_PORT,
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://${BACKEND_HOST}:${BACKEND_PORT}`,
        changeOrigin: true,
      },
      "/ws": {
        target: `ws://${BACKEND_HOST}:${BACKEND_PORT}`,
        ws: true,
      },
    },
  },
});
