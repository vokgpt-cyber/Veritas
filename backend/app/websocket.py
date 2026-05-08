"""WebSocket endpoints for real-time progress updates."""

import json
import logging
from typing import Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self._active_connections: Dict[str, Set[WebSocket]] = {}
        self._logger = logging.getLogger("ConnectionManager")

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Accept WebSocket connection and register it.

        Args:
            websocket: WebSocket connection
            job_id: Job identifier to subscribe to
        """
        await websocket.accept()

        if job_id not in self._active_connections:
            self._active_connections[job_id] = set()

        self._active_connections[job_id].add(websocket)
        self._logger.info(f"WebSocket connected for job {job_id}")

    async def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """
        Disconnect WebSocket and unregister it.

        Args:
            websocket: WebSocket connection
            job_id: Job identifier
        """
        if job_id in self._active_connections:
            self._active_connections[job_id].discard(websocket)

            if not self._active_connections[job_id]:
                del self._active_connections[job_id]

        self._logger.info(f"WebSocket disconnected for job {job_id}")

    async def broadcast_progress(self, job_id: str, data: dict) -> None:
        """
        Broadcast progress update to all connected clients for a job.

        Args:
            job_id: Job identifier
            data: Progress data dictionary
        """
        if job_id not in self._active_connections:
            return

        disconnected = set()

        for websocket in self._active_connections[job_id]:
            try:
                await websocket.send_json(data)
            except Exception as e:
                self._logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.add(websocket)

        # Remove disconnected connections
        for websocket in disconnected:
            await self.disconnect(websocket, job_id)

    def get_active_connections(self, job_id: str) -> int:
        """
        Get number of active connections for a job.

        Args:
            job_id: Job identifier

        Returns:
            Number of active connections
        """
        return len(self._active_connections.get(job_id, set()))


# Global connection manager
connection_manager = ConnectionManager()


@router.websocket("/ws/meetings/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time meeting progress updates.

    Path: /ws/meetings/{job_id}

    Message format:
    {
        "type": "progress" | "status" | "error",
        "job_id": "...",
        "progress": 0-100,
        "state": "uploading" | "preprocessing" | "transcribing" | ...,
        "current_stage": "...",
        "eta_seconds": float | null,
        "timestamp": ISO8601
    }

    Args:
        websocket: WebSocket connection
        job_id: Job identifier to subscribe to
    """
    await connection_manager.connect(websocket, job_id)

    try:
        orchestrator = None
        last_progress = -1

        while True:
            # Receive message from client (keep-alive)
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({"type": "pong"})

                elif message.get("type") == "status":
                    # Client requested status update
                    if orchestrator is None:
                        from backend.app.main import orchestrator as orch

                        orchestrator = orch

                    if orchestrator:
                        job = orchestrator.get_job(job_id)
                        if job:
                            await websocket.send_json({
                                "type": "status",
                                "job_id": job_id,
                                "progress": job.progress,
                                "state": job.state.value,
                                "current_stage": job.current_stage,
                                "eta_seconds": job.eta_seconds,
                                "timestamp": job.updated_at.isoformat(),
                            })

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket, job_id)
        logger.info(f"WebSocket client disconnected for job {job_id}")

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await connection_manager.disconnect(websocket, job_id)


async def send_progress_update(job_id: str) -> None:
    """
    Send progress update for a job to all connected clients.

    This function is called from the orchestrator's progress callback.

    Args:
        job_id: Job identifier
    """
    try:
        from backend.app.main import orchestrator

        if not orchestrator:
            return

        job = orchestrator.get_job(job_id)
        if not job:
            return

        # Prepare message
        message = {
            "type": "progress",
            "job_id": job_id,
            "progress": job.progress,
            "state": job.state.value,
            "current_stage": job.current_stage,
            "eta_seconds": job.eta_seconds,
            "timestamp": job.updated_at.isoformat(),
        }

        await connection_manager.broadcast_progress(job_id, message)

    except Exception as e:
        logger.error(f"Failed to send progress update for {job_id}: {e}")


# Export for use in main.py
app = router
