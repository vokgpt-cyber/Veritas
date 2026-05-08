"""Singleton GPU VRAM manager for monitoring and controlling GPU memory usage."""

import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class VRAMManager:
    """Singleton GPU memory manager. Tracks loaded models and monitors VRAM usage."""

    _instance: Optional["VRAMManager"] = None

    def __new__(cls) -> "VRAMManager":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize VRAM manager singleton."""
        if self._initialized:
            return
        self._initialized = True
        self._current_model: Optional[str] = None
        self._logger = logging.getLogger("VRAMManager")

    def get_status(self) -> dict:
        """
        Return GPU status dictionary.

        Returns:
            Dictionary with keys:
            - vram_used_gb: float
            - vram_total_gb: float
            - vram_free_gb: float
            - gpu_utilization: float (0-100%)
            - temperature: float (°C)
            - device_name: str
            - current_model: Optional[str]
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return self._get_dummy_status()

            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(device)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(device, 0)
            except Exception:
                temp = 0.0

            device_name = pynvml.nvmlDeviceGetName(device).decode("utf-8")

            pynvml.nvmlShutdown()

            return {
                "vram_used_gb": mem_info.used / (1024**3),
                "vram_total_gb": mem_info.total / (1024**3),
                "vram_free_gb": mem_info.free / (1024**3),
                "gpu_utilization": float(util.gpu),
                "temperature": float(temp),
                "device_name": device_name,
                "current_model": self._current_model,
            }

        except ImportError:
            self._logger.debug("pynvml not available, trying nvidia-smi")
            return self._get_status_from_nvidia_smi()
        except Exception as e:
            # Common on Windows: NVML shared library not on path.
            # Fall through to nvidia-smi — it ships with the driver and is
            # always on PATH if CUDA is usable at all. Log at debug to avoid
            # spamming (this method can be polled many times per second).
            self._logger.debug(
                f"pynvml failed ({e}), falling back to nvidia-smi"
            )
            return self._get_status_from_nvidia_smi()

    def _get_status_from_nvidia_smi(self) -> dict:
        """Get GPU status from nvidia-smi subprocess call."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,temperature.gpu,name",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return self._get_dummy_status()

            parts = result.stdout.strip().split(", ")
            if len(parts) < 4:
                return self._get_dummy_status()

            used_mb = float(parts[0])
            total_mb = float(parts[1])
            temp = float(parts[2])
            device_name = parts[3]

            return {
                "vram_used_gb": used_mb / 1024,
                "vram_total_gb": total_mb / 1024,
                "vram_free_gb": (total_mb - used_mb) / 1024,
                "gpu_utilization": 0.0,
                "temperature": temp,
                "device_name": device_name,
                "current_model": self._current_model,
            }

        except Exception as e:
            self._logger.debug(f"nvidia-smi failed: {e}")
            return self._get_dummy_status()

    def _get_dummy_status(self) -> dict:
        """Return dummy status when GPU is not available."""
        return {
            "vram_used_gb": 0.0,
            "vram_total_gb": 24.0,
            "vram_free_gb": 24.0,
            "gpu_utilization": 0.0,
            "temperature": 0.0,
            "device_name": "CPU",
            "current_model": self._current_model,
        }

    def check_available(self, required_gb: float) -> bool:
        """
        Check if enough VRAM is free for the required amount.

        Args:
            required_gb: Required VRAM in gigabytes

        Returns:
            True if sufficient VRAM is available, False otherwise
        """
        status = self.get_status()
        available = status["vram_free_gb"]
        result = available >= required_gb

        if not result:
            self._logger.warning(
                f"Insufficient VRAM: required {required_gb:.2f}GB, "
                f"available {available:.2f}GB"
            )

        return result

    async def wait_for_available(
        self, required_gb: float, timeout: float = 120.0
    ) -> bool:
        """
        Poll for available VRAM until sufficient amount is available or timeout.

        Args:
            required_gb: Required VRAM in gigabytes
            timeout: Maximum time to wait in seconds

        Returns:
            True if VRAM became available within timeout, False otherwise
        """
        import asyncio

        start_time = time.time()
        poll_interval = 2.0

        while time.time() - start_time < timeout:
            if self.check_available(required_gb):
                self._logger.info(f"VRAM available: {required_gb:.2f}GB")
                return True

            await asyncio.sleep(poll_interval)

        self._logger.error(
            f"Timeout waiting for {required_gb:.2f}GB VRAM after {timeout}s"
        )
        return False

    def release_all(self) -> None:
        """
        Release all GPU memory — run garbage collection and clear CUDA cache.

        Safe to call after engine.unload() has already freed model-specific
        GPU resources (e.g., CTranslate2 unload_model()). The gc.collect()
        and empty_cache() calls mop up any remaining PyTorch tensors and
        CUDA allocator blocks so that the next model has maximum VRAM.
        """
        self._current_model = None

        try:
            import gc
            gc.collect()
        except Exception as e:
            self._logger.debug(f"gc.collect() during release: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # NOTE: skip torch.cuda.synchronize() — it can deadlock if
                # CTranslate2 left pending CUDA operations in its own stream.
                # empty_cache() is sufficient to free the allocator blocks.
                self._logger.info("VRAM released: gc.collect() + CUDA cache cleared")
            else:
                self._logger.info("VRAM released (no CUDA device)")
        except ImportError:
            self._logger.info("VRAM released (torch not available)")
        except Exception as e:
            self._logger.warning(f"CUDA cache clear failed: {e}")
            self._logger.info("VRAM released (partial cleanup)")

    def register_model(self, name: str) -> None:
        """
        Register a model as currently loaded.

        Args:
            name: Model name/identifier
        """
        self._current_model = name
        self._logger.info(f"Registered model: {name}")

    def unregister_model(self) -> None:
        """Unregister the currently loaded model."""
        self._current_model = None
        self._logger.info("Unregistered current model")

    @property
    def is_gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model name."""
        return self._current_model
