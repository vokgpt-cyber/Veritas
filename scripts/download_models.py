"""
EPAM VERITAS — Model Download Script
Run this ONCE on a machine with internet access to cache all ML models locally.
After this, the pipeline works fully offline.

Usage:
    python scripts/download_models.py [--all | --whisper | --speechbrain | --pyannote | --summarization | --vad]
    python scripts/download_models.py --all          # download everything
    python scripts/download_models.py --whisper      # only Whisper ASR model
    python scripts/download_models.py --russian-whisper  # Russian-finetuned Whisper (recommended)

Requirements:
    pip install faster-whisper speechbrain transformers torch
    pip install pyannote.audio  # optional, only if using pyannote engine
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download_whisper(model_name: str = "large-v3") -> None:
    """Download faster-whisper model to local cache."""
    print(f"\n{'='*60}")
    print(f"Downloading faster-whisper model: {model_name}")
    print(f"{'='*60}")
    try:
        from faster_whisper import WhisperModel
        # This downloads the model to ~/.cache/huggingface/hub/
        print("Loading model (this will download if not cached)...")
        model = WhisperModel(model_name, device="cpu", compute_type="int8_float32")
        del model
        print(f"[OK] faster-whisper '{model_name}' cached successfully")
    except Exception as e:
        print(f"[FAIL] faster-whisper download failed: {e}")
        return


def download_russian_whisper() -> None:
    """Download Russian-finetuned faster-whisper model (35% better WER)."""
    download_whisper("bzikst/faster-whisper-large-v3-russian")


def download_speechbrain() -> None:
    """Download SpeechBrain ECAPA-TDNN speaker embedding model."""
    print(f"\n{'='*60}")
    print("Downloading SpeechBrain ECAPA-TDNN (speaker embeddings)")
    print(f"{'='*60}")
    try:
        from speechbrain.pretrained import EncoderClassifier
        print("Loading model (this will download if not cached)...")
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"},
        )
        del model
        print("[OK] SpeechBrain ECAPA-TDNN cached successfully")
    except Exception as e:
        print(f"[FAIL] SpeechBrain download failed: {e}")


def download_pyannote() -> None:
    """Download pyannote speaker diarization pipeline."""
    print(f"\n{'='*60}")
    print("Downloading pyannote speaker-diarization-3.1")
    print("NOTE: Requires HF_TOKEN with access to pyannote models")
    print(f"{'='*60}")
    try:
        from pyannote.audio import Pipeline
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[WARN] HF_TOKEN not set. pyannote models require HuggingFace token.")
            print("       Get token at https://huggingface.co/settings/tokens")
            print("       Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
            return
        print("Loading pipeline (this will download if not cached)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        del pipeline
        print("[OK] pyannote speaker-diarization-3.1 cached successfully")
    except ImportError:
        print("[SKIP] pyannote.audio not installed. Install with: pip install pyannote.audio")
    except Exception as e:
        print(f"[FAIL] pyannote download failed: {e}")


def download_summarization_gguf(
    repo: str = "Qwen/Qwen3-8B-GGUF",
    filename: str = "Qwen3-8B-Q4_K_M.gguf",
) -> None:
    """Download Qwen3-8B GGUF model for llama-cpp-python inference.

    Downloads Q4_K_M quantization (~4.9GB) to project models/ directory.
    This is a single file — no tokenizer or config needed (built into GGUF).
    """
    print(f"\n{'='*60}")
    print(f"Downloading GGUF model: {repo} / {filename}")
    print("This may take 5-10 minutes (~4.9GB). Be patient.")
    print(f"{'='*60}")

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    dest_path = models_dir / filename

    if dest_path.exists():
        size_gb = dest_path.stat().st_size / (1024**3)
        print(f"[OK] Already downloaded: {dest_path} ({size_gb:.2f} GB)")
        return

    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading to {dest_path} ...")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        size_gb = Path(downloaded).stat().st_size / (1024**3)
        print(f"[OK] GGUF model downloaded: {downloaded} ({size_gb:.2f} GB)")
    except ImportError:
        print("[FAIL] huggingface_hub not installed. Install with: pip install huggingface-hub")
        print(f"       Or manually download from https://huggingface.co/{repo}")
        print(f"       Place {filename} in {models_dir}/")
    except Exception as e:
        print(f"[FAIL] GGUF download failed: {e}")
        print(f"       You can manually download from https://huggingface.co/{repo}")
        print(f"       Place {filename} in {models_dir}/")


def download_vad() -> None:
    """Download Silero VAD model."""
    print(f"\n{'='*60}")
    print("Downloading Silero VAD (voice activity detection)")
    print(f"{'='*60}")
    try:
        import torch
        print("Loading model from torch hub...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        del model
        print("[OK] Silero VAD cached successfully")
    except Exception as e:
        print(f"[FAIL] Silero VAD download failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all ML models for EPAM VERITAS (offline preparation)"
    )
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--whisper", action="store_true", help="Download faster-whisper large-v3")
    parser.add_argument("--russian-whisper", action="store_true", help="Download Russian-finetuned Whisper (recommended)")
    parser.add_argument("--speechbrain", action="store_true", help="Download SpeechBrain ECAPA-TDNN")
    parser.add_argument("--pyannote", action="store_true", help="Download pyannote diarization")
    parser.add_argument("--summarization", action="store_true", help="Download Qwen3-8B GGUF (Q4_K_M)")
    parser.add_argument("--vad", action="store_true", help="Download Silero VAD")
    args = parser.parse_args()

    # If no flags, show help
    if not any([args.all, args.whisper, args.russian_whisper, args.speechbrain,
                args.pyannote, args.summarization, args.vad]):
        parser.print_help()
        print("\nRecommended for first-time setup:")
        print("  python scripts/download_models.py --all")
        sys.exit(0)

    print("EPAM VERITAS — Model Download")
    print(f"Cache directory: {Path.home() / '.cache' / 'huggingface'}")
    print(f"Torch hub cache: {Path.home() / '.cache' / 'torch'}")

    if args.all or args.vad:
        download_vad()
    if args.all or args.whisper:
        download_whisper("large-v3")
    if args.all or args.russian_whisper:
        download_russian_whisper()
    if args.all or args.speechbrain:
        download_speechbrain()
    if args.all or args.pyannote:
        download_pyannote()
    if args.all or args.summarization:
        download_summarization_gguf()

    print(f"\n{'='*60}")
    print("Done! Models are cached locally.")
    print("The pipeline will now work fully offline.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
