"""
EPAM VERITAS — Pre-flight Check
Run this BEFORE starting the pipeline to verify everything is ready.
Checks: Python, CUDA/GPU, dependencies, model cache, config, disk space, ports.

Usage:
    python scripts/preflight_check.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import socket
import sys
from pathlib import Path


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"

errors: list[str] = []
warnings: list[str] = []


def check(label: str, ok: bool, fail_msg: str = "", warn_only: bool = False) -> bool:
    if ok:
        print(f"  {PASS} {label}")
    elif warn_only:
        print(f"  {WARN} {label} -- {fail_msg}")
        warnings.append(f"{label}: {fail_msg}")
    else:
        print(f"  {FAIL} {label} -- {fail_msg}")
        errors.append(f"{label}: {fail_msg}")
    return ok


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_python() -> None:
    section("Python Environment")
    v = sys.version_info
    check("Python version", v.major == 3 and v.minor >= 10,
          f"Need Python 3.10+, got {v.major}.{v.minor}.{v.micro}")
    check("Platform", True)
    print(f"         {sys.platform} / {os.name}")


def check_gpu() -> None:
    section("GPU / CUDA")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        check("PyTorch installed", True)
        check("CUDA available", has_cuda, "No CUDA — will use CPU (very slow)", warn_only=True)
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            check(f"GPU: {gpu_name}", True)
            check(f"VRAM: {vram_mb:.0f} MB", vram_mb >= 6000,
                  f"Need 6+ GB VRAM for full pipeline, got {vram_mb:.0f} MB")
    except ImportError:
        check("PyTorch installed", False, "pip install torch")


def check_dependencies() -> None:
    section("Python Dependencies")
    required = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("yaml", "PyYAML"),
        ("jose", "python-jose[cryptography]"),
        ("passlib", "passlib[bcrypt]"),
        ("faster_whisper", "faster-whisper"),
        ("speechbrain", "speechbrain"),
        ("sklearn", "scikit-learn"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("docx", "python-docx"),
        ("cryptography", "cryptography"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
    ]
    optional = [
        ("pyannote.audio", "pyannote.audio"),
        ("bitsandbytes", "bitsandbytes"),
        ("nemo_toolkit", "nemo_toolkit[asr]"),
    ]

    for module, pip_name in required:
        try:
            importlib.import_module(module)
            check(f"{pip_name}", True)
        except ImportError:
            check(f"{pip_name}", False, f"pip install {pip_name}")

    print("\n  Optional:")
    for module, pip_name in optional:
        try:
            importlib.import_module(module)
            check(f"{pip_name}", True)
        except ImportError:
            check(f"{pip_name}", False, f"pip install {pip_name}", warn_only=True)


def check_models() -> None:
    section("Model Cache (offline readiness)")
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    torch_cache = Path.home() / ".cache" / "torch" / "hub"

    # Silero VAD
    vad_cached = any(torch_cache.glob("snakers4_silero-vad*")) if torch_cache.exists() else False
    check("Silero VAD", vad_cached, "Run: python scripts/download_models.py --vad")

    # faster-whisper
    whisper_dirs = list(hf_cache.glob("models--Systran--faster-whisper-large-v3*")) if hf_cache.exists() else []
    check("faster-whisper large-v3", len(whisper_dirs) > 0,
          "Run: python scripts/download_models.py --whisper")

    # Russian Whisper (optional but recommended)
    ru_whisper = list(hf_cache.glob("models--bzikst--faster-whisper-large-v3-russian*")) if hf_cache.exists() else []
    check("Russian-finetuned Whisper", len(ru_whisper) > 0,
          "Run: python scripts/download_models.py --russian-whisper", warn_only=True)

    # SpeechBrain
    sb_cache = Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
    sb_alt = list(hf_cache.glob("models--speechbrain--spkrec-ecapa-voxceleb*")) if hf_cache.exists() else []
    check("SpeechBrain ECAPA-TDNN", sb_cache.exists() or len(sb_alt) > 0,
          "Run: python scripts/download_models.py --speechbrain")

    # pyannote (optional)
    pya_dirs = list(hf_cache.glob("models--pyannote--speaker-diarization*")) if hf_cache.exists() else []
    check("pyannote diarization", len(pya_dirs) > 0,
          "Run: python scripts/download_models.py --pyannote (optional)", warn_only=True)

    # Qwen3-8B
    qwen3_dirs = list(hf_cache.glob("models--Qwen--Qwen3-8B*")) if hf_cache.exists() else []
    qwen25_dirs = list(hf_cache.glob("models--Qwen--Qwen2.5-7B*")) if hf_cache.exists() else []
    check("Qwen3-8B (summarization)", len(qwen3_dirs) > 0,
          "Run: python scripts/download_models.py --summarization")
    if len(qwen25_dirs) > 0 and len(qwen3_dirs) == 0:
        print(f"  {WARN} Found old Qwen2.5-7B but not Qwen3-8B. Upgrade recommended.")


def check_config() -> None:
    section("Configuration")
    config_path = Path("config/settings.yaml")
    check("config/settings.yaml exists", config_path.exists(), "Missing config file!")

    env_path = Path(".env")
    check(".env file exists", env_path.exists(), "Copy .env.example to .env and fill in secrets")

    if env_path.exists():
        env_content = env_path.read_text()
        check("AUTH_SECRET_KEY set", "CHANGE_ME" not in env_content.split("AUTH_SECRET_KEY")[1][:50] if "AUTH_SECRET_KEY" in env_content else False,
              "Generate a real secret key!")
        check("ENCRYPTION_SECRET_KEY set", "CHANGE_ME" not in env_content.split("ENCRYPTION_SECRET_KEY")[1][:50] if "ENCRYPTION_SECRET_KEY" in env_content else False,
              "Generate a real encryption key!")


def check_ffmpeg() -> None:
    section("External Tools")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    check("ffmpeg installed", ffmpeg is not None,
          "Install ffmpeg (torchaudio fallback available but slower)", warn_only=True)
    check("ffprobe installed", ffprobe is not None,
          "Install ffprobe (torchaudio fallback available but slower)", warn_only=True)

    if ffprobe:
        # Check if ffprobe supports -print_json
        import subprocess
        try:
            result = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", ffprobe],
                capture_output=True, timeout=5
            )
            check("ffprobe -print_json support", result.returncode == 0 or b"json" in result.stderr,
                  "Old ffprobe version, torchaudio fallback will be used", warn_only=True)
        except Exception:
            pass

    node = shutil.which("node")
    npm = shutil.which("npm")
    check("Node.js installed (for frontend)", node is not None,
          "Install Node.js 18+ for frontend", warn_only=True)


def check_disk() -> None:
    section("Disk Space")
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    check(f"Free disk space: {free_gb:.1f} GB", free_gb >= 10,
          f"Need 10+ GB free (models + audio + outputs), got {free_gb:.1f} GB")

    # Check data directories
    for d in ["data/meetings", "data/archive", "data/audit"]:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"  {PASS} Created {d}/")
        else:
            print(f"  {PASS} {d}/ exists")


def check_ports() -> None:
    section("Network Ports")
    for port, name in [(8000, "Backend API"), (3000, "Frontend dev server")]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result == 0:
            check(f"Port {port} ({name})", False,
                  f"Port {port} already in use! Stop the process or change port.", warn_only=True)
        else:
            check(f"Port {port} ({name}) available", True)


def main() -> None:
    print("\nEPAM VERITAS — Pre-flight Check")
    print("=" * 60)

    check_python()
    check_gpu()
    check_dependencies()
    check_models()
    check_config()
    check_ffmpeg()
    check_disk()
    check_ports()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    if errors:
        print(f"\n  {FAIL} {len(errors)} error(s) must be fixed:")
        for e in errors:
            print(f"       - {e}")
    if warnings:
        print(f"\n  {WARN} {len(warnings)} warning(s) (non-blocking):")
        for w in warnings:
            print(f"       - {w}")
    if not errors:
        print(f"\n  {PASS} All critical checks passed! Ready for on-premise test.")
    else:
        print(f"\n  {FAIL} Fix the errors above before running the pipeline.")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
