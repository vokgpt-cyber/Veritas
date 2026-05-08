"""Build clean user-facing deliverables (transcript.md + protocol.docx/json) in
the `deliverables/` folder, from the raw output of a completed pipeline run.

Applies output-cleaning rules that the orchestrator does not (yet) apply:
  * Strip chat-template markers from the title (<|im_end|>, <|endoftext|>)
  * Drop separator/placeholder items ("--", empty) from list sections
  * Drop verification-pass commentary leaked into open_questions

Usage:
    python scripts/build_deliverables.py <job_id> [--base NAME]

If --base is omitted, uses the audio filename (sanitized to ASCII).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLES = REPO_ROOT / "deliverables"


# ---------- cleaning helpers ----------

_CHAT_MARKERS = [
    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
    "<|assistant|>", "<|user|>", "<|system|>",
]


def clean_text(s: str) -> str:
    if not s:
        return ""
    out = s
    for m in _CHAT_MARKERS:
        out = out.replace(m, "")
    # Also strip any thinking tags that slipped through
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    out = re.sub(r"<think>.*", "", out, flags=re.DOTALL)
    return out.strip()


def is_noise_item(text: str) -> bool:
    s = text.strip()
    if not s or s in ("--", "---", "—", "–", "-", "**", "***"):
        return True
    # Verification-pass commentary that sometimes leaks into the list:
    noise_starts = (
        "**В разделе ", "*Примечание", "Примечание:", "Упоминание \"",
        "Note:", "*Note", "**Note",
    )
    if any(s.startswith(p) for p in noise_starts):
        return True
    return False


def clean_list(items: list, text_keys: tuple = ("title", "text")) -> list:
    cleaned = []
    for it in items:
        if isinstance(it, str):
            if not is_noise_item(it):
                cleaned.append(clean_text(it))
            continue
        if not isinstance(it, dict):
            continue
        text = ""
        for k in text_keys:
            if it.get(k):
                text = it[k]
                break
        if is_noise_item(text):
            continue
        it = dict(it)
        for k in text_keys:
            if k in it and it[k]:
                it[k] = clean_text(it[k])
        cleaned.append(it)
    return cleaned


def sanitize_basename(s: str) -> str:
    """Convert an arbitrary filename (incl. Cyrillic) into ASCII-safe base."""
    # Strip extension
    base = Path(s).stem
    # Transliterate Cyrillic letters to Latin approximations for filenames
    cyr = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
    lat = ["a","b","v","g","d","e","zh","z","i","y","k","l","m","n","o","p","r",
           "s","t","u","f","h","c","ch","sh","sch","","y","","e","yu","ya"]
    cmap = dict(zip(cyr, lat))
    cmap.update({c.upper(): v.capitalize() for c, v in cmap.items()})
    out = "".join(cmap.get(ch, ch) for ch in base)
    # Replace spaces and other non-ASCII with underscore
    out = re.sub(r"[^\w\-.]", "_", out, flags=re.ASCII)
    # Collapse runs of _
    out = re.sub(r"_+", "_", out).strip("_.")
    return out or "meeting"


# ---------- transcript markdown ----------

def format_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}"


def build_transcript_md(
    aligned_segments: list,
    base: str,
    audio_name: str,
    asr_engine: str = "GigaAM v3_e2e_rnnt",
    diarization_engine: str = "pyannote 4.0 community-1",
) -> str:
    lines: list[str] = []
    lines.append(f"# Transcript — {base.replace('_', ' ')}")
    lines.append("")
    lines.append(f"**Audio**: `{audio_name}`")
    duration = float(aligned_segments[-1].get("end", 0.0)) if aligned_segments else 0.0
    lines.append(f"**Duration**: {format_timestamp(duration)} ({duration/60:.1f} min)")
    speakers = {s.get("speaker_id") or "?" for s in aligned_segments}
    lines.append(f"**Segments**: {len(aligned_segments)}  •  **Speakers**: {len(speakers)}")
    lines.append(f"**ASR**: {asr_engine}  •  **Diarization**: {diarization_engine}")
    lines.append("")
    lines.append("---")
    lines.append("")

    cur_spk: str | None = None
    for seg in aligned_segments:
        spk = seg.get("speaker_id") or seg.get("speaker") or "?"
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = clean_text(seg.get("text") or "")
        if not text:
            continue
        if spk != cur_spk:
            lines.append("")
            lines.append(f"### {spk}  `[{format_timestamp(start)} – {format_timestamp(end)}]`")
            cur_spk = spk
        else:
            lines.append(f"_([{format_timestamp(start)} – {format_timestamp(end)}])_")
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


# ---------- main ----------

def build(job_id: str, base_override: str | None = None) -> dict:
    meeting_dir = REPO_ROOT / "data" / "meetings" / job_id
    if not meeting_dir.exists():
        raise SystemExit(f"ERROR: meeting dir not found: {meeting_dir}")

    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    # --- protocol JSON: clean and copy ---
    src_json = meeting_dir / f"{job_id}.json"
    if not src_json.exists():
        raise SystemExit(f"ERROR: protocol JSON not found: {src_json}")
    protocol = json.loads(src_json.read_text(encoding="utf-8"))

    protocol["topic"] = clean_text(protocol.get("topic") or "")
    protocol["summary"] = clean_text(protocol.get("summary") or "")
    protocol["key_topics"] = clean_list(protocol.get("key_topics", []))
    protocol["decisions"] = clean_list(protocol.get("decisions", []))
    protocol["tasks"] = clean_list(protocol.get("tasks", []))
    protocol["open_questions"] = clean_list(protocol.get("open_questions", []))

    # Pick deliverable base name
    audio_name = ""
    for p in (REPO_ROOT / "data" / "meetings" / job_id).iterdir():
        if p.suffix in (".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac") and "_processed" in p.name:
            audio_name = p.name.replace("_processed", "")
            break
    if base_override:
        base = base_override
    elif audio_name:
        base = sanitize_basename(audio_name)
    else:
        base = sanitize_basename(protocol.get("topic") or job_id)

    dst_json = DELIVERABLES / f"{base}_PROTOCOL.json"
    dst_json.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- protocol DOCX: copy as-is (cleaner regeneration would need orchestrator) ---
    src_docx = meeting_dir / f"{job_id}.docx"
    dst_docx = DELIVERABLES / f"{base}_PROTOCOL.docx"
    if src_docx.exists():
        shutil.copy2(src_docx, dst_docx)

    # --- transcript markdown ---
    aligned_path = meeting_dir / "aligned.json"
    if aligned_path.exists():
        aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
        segs = aligned if isinstance(aligned, list) else aligned.get("segments", [])
        md = build_transcript_md(segs, base, audio_name or job_id)
        dst_md = DELIVERABLES / f"{base}_TRANSCRIPT.md"
        dst_md.write_text(md, encoding="utf-8")
    else:
        dst_md = None

    return {
        "base": base,
        "json": str(dst_json),
        "docx": str(dst_docx) if src_docx.exists() else None,
        "md": str(dst_md) if dst_md else None,
        "protocol_summary": {
            "topic": protocol["topic"][:80],
            "summary_chars": len(protocol["summary"]),
            "participants": len(protocol.get("participants", [])),
            "key_topics": len(protocol["key_topics"]),
            "decisions": len(protocol["decisions"]),
            "tasks": len(protocol["tasks"]),
            "open_questions": len(protocol["open_questions"]),
            "transcript_segments": len(protocol.get("transcript", [])),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build clean deliverables from a finished pipeline run")
    ap.add_argument("job_id", help="Job UUID (sub-folder name under data/meetings/)")
    ap.add_argument("--base", default=None, help="Base filename for deliverables (ASCII-safe)")
    args = ap.parse_args()

    result = build(args.job_id, args.base)

    print("=" * 60)
    print(f"Deliverables written to: {DELIVERABLES}")
    print("=" * 60)
    for k in ("md", "docx", "json"):
        v = result.get(k)
        if v:
            print(f"  {k.upper():4s}: {v}")
    print()
    print("Protocol summary:")
    for k, v in result["protocol_summary"].items():
        print(f"  {k:20s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
