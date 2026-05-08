"""Diarization benchmark runner — Court hearing 129.

Runs each candidate diarizer on the same audio, emits:
    benchmarks/diar/<engine>/court_hearing_129.rttm
    benchmarks/diar/<engine>/court_hearing_129_run.json    (timing, VRAM, seg count, speakers)
    benchmarks/diar/<engine>/court_hearing_129_segments.json

Usage:
    python scripts/run_diar_benchmark.py \\
        --audio "data/test/Court hearing 129_01.mp4" \\
        --engines pyannote_community1 pyannote_31 speechbrain

    # Or run everything that's installed:
    python scripts/run_diar_benchmark.py --audio "data/test/Court hearing 129_01.mp4" --all

Engines:
    pyannote_community1  — pyannote-audio 4.0, speaker-diarization-community-1
    pyannote_31          — pyannote-audio 4.0 loading pyannote/speaker-diarization-3.1
    speechbrain          — ECAPA-TDNN + spectral clustering (existing fallback engine)
    nemo_sortformer      — NVIDIA Sortformer 4spk v1 (requires nemo_toolkit)
    nemo_msdd            — NVIDIA MSDD telephonic (requires nemo_toolkit)

Speaker hints:
    Gold says 4 speakers. All engines receive min_speakers=2, max_speakers=8 so
    they aren't handed the answer.  Measured DER against a 4-speaker reference.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Load .env so HF_TOKEN and other secrets are available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass


MIN_SPEAKERS_HINT = 2
MAX_SPEAKERS_HINT = 8


def _peak_vram_gb() -> float:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        pass
    return 0.0


def _reset_vram_peak() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_audio_waveform(audio_path: Path):
    """Load audio to mono waveform via torchaudio (bypasses torchcodec).
    Passes native sample rate through; pyannote handles resampling internally.
    Returns (waveform_tensor [1, T], sample_rate).
    """
    import torchaudio  # type: ignore

    wav, sr = torchaudio.load(str(audio_path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def _ensure_wav(audio_path: Path) -> Path:
    """Preprocess MP4/other formats to 16 kHz mono WAV via ffmpeg.

    torchcodec was uninstalled (Bug 59) because it's incompatible with
    torch 2.8.0 on Windows, which means torchaudio's default backend
    (soundfile) can only read WAV/FLAC/OGG.  Everything else needs an
    ffmpeg pre-pass.

    Returns the path to a 16 kHz mono WAV. If the input is already a WAV,
    it's returned as-is.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    wav_dir = REPO_ROOT / "benchmarks" / "audio_cache"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / (audio_path.stem + "_16k_mono.wav")

    if wav_path.is_file() and wav_path.stat().st_mtime > audio_path.stat().st_mtime:
        print(f"using cached WAV: {wav_path}")
        return wav_path

    print(f"ffmpeg preprocess: {audio_path.name} -> {wav_path.name}")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )
    if not wav_path.is_file():
        raise RuntimeError(f"ffmpeg reported success but no output at {wav_path}")
    print(f"  wrote {wav_path.stat().st_size / (1024*1024):.1f} MB WAV")
    return wav_path


def run_pyannote(
    audio_path: Path,
    model_name: str,
    device: str = "cuda",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """Generic pyannote-audio pipeline runner.
    Handles both 3.x Annotation and 4.x DiarizeOutput shapes.
    """
    import torch  # type: ignore
    from pyannote.audio import Pipeline  # type: ignore

    t_load = time.time()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    load_time = time.time() - t_load

    # Pre-load audio waveform to avoid torchcodec dependency (Bug 59)
    wav, sr = _load_audio_waveform(audio_path)
    audio_input = {"waveform": wav, "sample_rate": sr}

    t_run = time.time()
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    raw = pipeline(audio_input, **kwargs)
    run_time = time.time() - t_run

    # Normalise to pyannote Annotation
    if hasattr(raw, "speaker_diarization"):
        annot = raw.speaker_diarization
    else:
        annot = raw

    segments = []
    speakers = set()
    for segment, _, label in annot.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(label),
            }
        )
        speakers.add(str(label))

    return {
        "load_time_s": round(load_time, 3),
        "run_time_s": round(run_time, 3),
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "segments": segments,
    }


def run_speechbrain(audio_path: Path, num_speakers: int | None = None):
    """Use the backend's SpeechBrain diarization engine for the benchmark."""
    import asyncio

    from backend.app.config import AppConfig  # type: ignore
    from backend.engine.diarization import DiarizationEngine  # type: ignore

    cfg = AppConfig()
    engine = DiarizationEngine(cfg)

    t_load = time.time()
    engine.load()
    load_time = time.time() - t_load

    # process() is async — run it in a temporary event loop
    t_run = time.time()
    result = asyncio.run(
        engine.process(
            audio_path=str(audio_path),
            min_speakers=MIN_SPEAKERS_HINT,
            max_speakers=MAX_SPEAKERS_HINT,
        )
    )
    run_time = time.time() - t_run
    engine.unload()

    # result is list of DiarizationSegment (start, end, speaker_id).
    # Pydantic field is `speaker_id` (see backend/app/models.py); some older
    # code paths also expose `speaker` as an alias — try both.
    segments = []
    speakers = set()
    for s in result:
        if hasattr(s, "start"):
            spk = getattr(s, "speaker_id", None) or getattr(s, "speaker", None) or "SPK?"
            seg = {"start": float(s.start), "end": float(s.end), "speaker": str(spk)}
        else:
            spk = s.get("speaker_id") or s.get("speaker") or "SPK?"
            seg = {"start": float(s["start"]), "end": float(s["end"]), "speaker": str(spk)}
        segments.append(seg)
        speakers.add(seg["speaker"])

    return {
        "load_time_s": round(load_time, 3),
        "run_time_s": round(run_time, 3),
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "segments": segments,
    }


def run_nemo_sortformer(audio_path: Path):
    """NVIDIA Sortformer 4-speaker v1 via nemo_toolkit."""
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"nemo_toolkit not installed: {e}")

    t_load = time.time()
    # Downloads from HF on first run
    model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    try:
        model.cuda()
    except Exception:
        pass
    model.eval()
    load_time = time.time() - t_load

    t_run = time.time()
    # Sortformer inference: .diarize accepts a list of audio paths
    result = model.diarize(audio=str(audio_path), batch_size=1)
    run_time = time.time() - t_run

    # result is a list (one per input audio), each element is a list of
    # "start end speaker" strings.
    first = result[0] if isinstance(result, list) else result
    segments = []
    speakers = set()
    for line in first:
        parts = line.strip().split()
        if len(parts) >= 3:
            start = float(parts[0])
            end = float(parts[1])
            spk = " ".join(parts[2:])
            segments.append({"start": start, "end": end, "speaker": spk})
            speakers.add(spk)

    return {
        "load_time_s": round(load_time, 3),
        "run_time_s": round(run_time, 3),
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "segments": segments,
    }


def run_nemo_msdd(audio_path: Path):
    """NVIDIA MSDD (Multiscale Diarization Decoder) via nemo_toolkit.
    Uses the telephonic preset by default.
    """
    try:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"nemo_toolkit not installed: {e}")

    import tempfile
    from omegaconf import OmegaConf  # type: ignore

    # Build a minimal NeMo config on the fly pointing at our single audio file
    work_dir = Path(tempfile.mkdtemp(prefix="nemo_msdd_"))
    manifest_path = work_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "audio_filepath": str(audio_path.resolve()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = OmegaConf.create(
        {
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(work_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": 0.5,
                        "offset": 0.3,
                        "pad_onset": 0.1,
                        "pad_offset": 0.1,
                        "min_duration_on": 0.2,
                        "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    }
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "use_speaker_model_from_ckpt": True,
                        "infer_batch_size": 25,
                        "sigmoid_threshold": [0.7],
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "diar_window_length": 50,
                        "overlap_infer_spk_limit": 5,
                    },
                },
            }
        }
    )

    t_load = time.time()
    diarizer = NeuralDiarizer(cfg=cfg)
    load_time = time.time() - t_load

    t_run = time.time()
    diarizer.diarize()
    run_time = time.time() - t_run

    # NeuralDiarizer writes RTTM into work_dir/pred_rttms/<audio_stem>.rttm
    rttm_candidates = list((work_dir / "pred_rttms").glob("*.rttm"))
    if not rttm_candidates:
        raise RuntimeError("MSDD produced no RTTM output")
    rttm = rttm_candidates[0]
    segments = []
    speakers = set()
    for line in rttm.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts or parts[0] != "SPEAKER":
            continue
        # SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
        start = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        segments.append({"start": start, "end": start + dur, "speaker": spk})
        speakers.add(spk)

    return {
        "load_time_s": round(load_time, 3),
        "run_time_s": round(run_time, 3),
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "segments": segments,
    }


# ---------- Registry ----------
ENGINES = {
    "pyannote_community1": {
        "display": "pyannote 4.0 community-1",
        "folder": "pyannote_community1",
        "runner": lambda audio: run_pyannote(
            audio,
            "pyannote/speaker-diarization-community-1",
            min_speakers=MIN_SPEAKERS_HINT,
            max_speakers=MAX_SPEAKERS_HINT,
        ),
    },
    "pyannote_31": {
        "display": "pyannote 3.1",
        "folder": "pyannote_31",
        "runner": lambda audio: run_pyannote(
            audio,
            "pyannote/speaker-diarization-3.1",
            min_speakers=MIN_SPEAKERS_HINT,
            max_speakers=MAX_SPEAKERS_HINT,
        ),
    },
    "speechbrain": {
        "display": "SpeechBrain ECAPA-TDNN",
        "folder": "speechbrain",
        "runner": lambda audio: run_speechbrain(audio),
    },
    "nemo_sortformer": {
        "display": "NVIDIA Sortformer 4spk v1",
        "folder": "nemo_sortformer",
        "runner": lambda audio: run_nemo_sortformer(audio),
    },
    "nemo_msdd": {
        "display": "NVIDIA NeMo MSDD telephonic",
        "folder": "nemo_msdd",
        "runner": lambda audio: run_nemo_msdd(audio),
    },
}


def _safe_speaker_label(label: str) -> str:
    """RTTM is whitespace-separated; speaker labels with spaces break the
    pandas-based RTTM loader in pyannote.database. Collapse all whitespace."""
    return "_".join(str(label).split())


def _write_rttm(segments: list[dict], path: Path, recording_id: str) -> None:
    lines = []
    for s in segments:
        dur = max(0.05, s["end"] - s["start"])
        spk = _safe_speaker_label(s["speaker"])
        lines.append(
            f"SPEAKER {recording_id} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plain(segments: list[dict], path: Path) -> None:
    lines = []
    for s in segments:
        lines.append(f"[{s['start']:7.2f} → {s['end']:7.2f}] {s['speaker']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_engine(
    key: str,
    audio: Path,
    out_root: Path,
    recording_id: str,
) -> dict:
    spec = ENGINES[key]
    out_dir = out_root / spec["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    _reset_vram_peak()
    gc.collect()

    entry = {
        "engine": key,
        "display": spec["display"],
        "audio": str(audio),
        "recording_id": recording_id,
    }
    t0 = time.time()
    try:
        print(f"\n=== {spec['display']} ===")
        result = spec["runner"](audio)
        wall = time.time() - t0
        peak_vram = _peak_vram_gb()
        entry.update(result)
        entry["wall_time_s"] = round(wall, 3)
        entry["vram_peak_gb"] = round(peak_vram, 3)
        entry["status"] = "ok"

        # Write artefacts
        _write_rttm(result["segments"], out_dir / f"{recording_id}.rttm", recording_id)
        _write_plain(result["segments"], out_dir / f"{recording_id}.plain.txt")
        (out_dir / f"{recording_id}_segments.json").write_text(
            json.dumps(result["segments"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  ok: {result['n_segments']} segs, {result['n_speakers']} speakers, "
            f"{wall:.1f}s wall, {peak_vram:.2f} GB VRAM peak"
        )
    except Exception as e:
        entry["status"] = "failed"
        entry["error"] = str(e)
        entry["traceback"] = traceback.format_exc()
        print(f"  FAILED: {e}")
        print(traceback.format_exc())

    (out_dir / f"{recording_id}_run.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Best-effort cleanup between engines
    try:
        import torch  # type: ignore

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return entry


def _score_engine(
    engine_key: str,
    hyp_rttm: Path,
    ref_rttm: Path,
    metrics_dir: Path,
    collar: float,
) -> dict | None:
    try:
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "scripts"))
        import score_der  # type: ignore

        ref = score_der.load_rttm_annotation(ref_rttm)
        hyp = score_der.load_rttm_annotation(hyp_rttm)
        metrics = score_der.score(ref, hyp, collar)
        metrics["tag"] = engine_key
        metrics["ref_file"] = str(ref_rttm)
        metrics["hyp_file"] = str(hyp_rttm)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = metrics_dir / f"diar_{engine_key}_court_hearing_129.json"
        out.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metrics
    except Exception as e:
        print(f"  score failed for {engine_key}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--engines", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "benchmarks" / "diar")
    ap.add_argument("--recording-id", default="court_hearing_129")
    ap.add_argument("--skip-score", action="store_true",
                    help="Skip DER scoring after each engine")
    ap.add_argument("--ref-rttm", type=Path,
                    default=REPO_ROOT / "benchmarks" / "gold" / "court_hearing_129.rttm")
    ap.add_argument("--collar", type=float, default=0.25,
                    help="DER forgiveness collar in seconds")
    args = ap.parse_args()

    if not args.audio.is_file():
        raise SystemExit(f"audio not found: {args.audio}")

    # torchcodec was removed (Bug 59), so non-WAV formats must be pre-converted
    # via ffmpeg before hitting torchaudio / pyannote / speechbrain.
    audio_path = _ensure_wav(args.audio)
    print(f"using audio: {audio_path}\n")

    if args.all:
        keys = list(ENGINES.keys())
    elif args.engines:
        keys = args.engines
    else:
        keys = ["pyannote_community1", "pyannote_31", "speechbrain"]

    unknown = [k for k in keys if k not in ENGINES]
    if unknown:
        raise SystemExit(f"unknown engines: {unknown}. known: {list(ENGINES)}")

    summary = []
    metrics_dir = REPO_ROOT / "benchmarks" / "metrics"

    for key in keys:
        entry = run_engine(key, audio_path, args.out_root, args.recording_id)
        # Auto-score if successful and reference exists
        if (
            not args.skip_score
            and entry["status"] == "ok"
            and args.ref_rttm.is_file()
        ):
            hyp_rttm = args.out_root / ENGINES[key]["folder"] / f"{args.recording_id}.rttm"
            if hyp_rttm.is_file():
                scored = _score_engine(key, hyp_rttm, args.ref_rttm, metrics_dir, args.collar)
                if scored is not None:
                    entry["der"] = scored["der"]
                    entry["der_skip_overlap"] = scored["der_skip_overlap"]
                    entry["jer"] = scored["jer"]
                    entry["miss_s"] = scored["miss"]
                    entry["false_alarm_s"] = scored["false_alarm"]
                    entry["confusion_s"] = scored["confusion"]
        summary.append(entry)

    print("\n=== Summary ===")
    header = f"  {'Engine':<32s} {'Segs':>6s} {'Spk':>4s} {'Wall':>8s} {'VRAM':>7s} {'DER':>8s} {'JER':>8s}"
    print(header)
    for e in summary:
        if e["status"] == "ok":
            der = f"{e.get('der', 0) * 100:.2f}%" if "der" in e else "n/a"
            jer = f"{e.get('jer', 0) * 100:.2f}%" if "jer" in e else "n/a"
            print(
                f"  {e['display']:<32s} "
                f"{e['n_segments']:>6d} "
                f"{e['n_speakers']:>4d} "
                f"{e['wall_time_s']:>7.1f}s "
                f"{e['vram_peak_gb']:>6.2f}G "
                f"{der:>8s} "
                f"{jer:>8s}"
            )
        else:
            print(f"  {e['display']:<32s} FAILED: {e.get('error')}")


if __name__ == "__main__":
    main()
