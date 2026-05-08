"""Synthesize a reference RTTM for Court hearing 129 by aligning gold turns to
GigaAM v3 word-level timestamps.

Why: the gold jsonl has (turn, speaker, text) but NO timestamps.  DER scoring
needs time-aligned speaker spans. GigaAM v3 emits word-level timestamps for the
full audio with 26 % WER — good enough to carry gold turns into the time axis
via monotonic token alignment.

Algorithm:
    1. Normalize + tokenize both sides (lowercase, ё→е, strip punct).
    2. Flatten gold into a single token stream, remembering which turn each
       token belongs to.
    3. Use difflib.SequenceMatcher to find matching blocks between the two
       token streams.
    4. For each gold turn, take the earliest GigaAM start and latest GigaAM end
       among tokens that aligned to tokens in this turn.  Fall back to
       interpolation between neighbours if a turn has no matches (short
       confirmation turns like "Да.").
    5. Clip overlapping turn boundaries so adjacent turns don't overlap.
    6. Emit RTTM and a debug JSON with per-turn alignment stats.

Output:
    benchmarks/gold/court_hearing_129.rttm
    benchmarks/gold/court_hearing_129_alignment.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_JSONL = ROOT / "benchmarks" / "gold" / "court_hearing_129.jsonl"
GIGAAM_SEGMENTS = ROOT / "benchmarks" / "asr" / "gigaam_v3" / "court_hearing_129_segments.json"
OUT_RTTM = ROOT / "benchmarks" / "gold" / "court_hearing_129.rttm"
OUT_DEBUG = ROOT / "benchmarks" / "gold" / "court_hearing_129_alignment.json"
RECORDING_ID = "court_hearing_129"
TOTAL_DURATION = 3332.11


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace("\u0451", "\u0435")  # ё -> е
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def tokens_of(text: str) -> list[str]:
    return normalise(text).split()


def load_gold() -> list[dict]:
    turns = []
    for line in GOLD_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            turns.append(json.loads(line))
    return turns


def load_gigaam_words() -> list[dict]:
    data = json.load(GIGAAM_SEGMENTS.open("r", encoding="utf-8"))
    words = []
    for seg in data:
        for w in seg.get("words", []):
            tok = normalise(w.get("word", ""))
            if tok:
                # Word may have a leading "—" or similar already stripped by normalise.
                for sub in tok.split():
                    words.append(
                        {
                            "token": sub,
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                        }
                    )
    return words


def align_gold_to_gigaam(gold_turns: list[dict], gigaam: list[dict]) -> list[dict]:
    # Flatten gold into (turn_idx, token)
    flat_gold: list[tuple[int, str]] = []
    for ti, turn in enumerate(gold_turns):
        for tok in tokens_of(turn.get("text", "")):
            flat_gold.append((ti, tok))

    gold_tok_seq = [t for _, t in flat_gold]
    gigaam_tok_seq = [w["token"] for w in gigaam]

    print(f"gold tokens: {len(gold_tok_seq)}")
    print(f"gigaam tokens: {len(gigaam_tok_seq)}")

    # Monotonic alignment via SequenceMatcher — O(N*M) worst-case but much
    # better in practice thanks to autojunk and matching-block heuristics.
    sm = SequenceMatcher(a=gold_tok_seq, b=gigaam_tok_seq, autojunk=False)
    blocks = sm.get_matching_blocks()  # last block has size 0, terminator
    print(f"matching blocks: {len(blocks)-1}")

    # Build per-gold-token -> gigaam idx map (only for matched tokens)
    gold_to_gigaam: dict[int, int] = {}
    matched = 0
    for block in blocks:
        a, b, size = block.a, block.b, block.size
        for k in range(size):
            gold_to_gigaam[a + k] = b + k
            matched += 1
    print(f"aligned tokens: {matched} ({matched / max(1, len(gold_tok_seq)) * 100:.1f}%)")

    # For each gold turn, gather gigaam start/end from matched tokens
    per_turn: list[dict] = []
    # Build an inverse mapping turn_idx -> list of gold_tok_idx
    turn_tok_indices: dict[int, list[int]] = {}
    for i, (ti, _) in enumerate(flat_gold):
        turn_tok_indices.setdefault(ti, []).append(i)

    for ti, turn in enumerate(gold_turns):
        tok_ids = turn_tok_indices.get(ti, [])
        matched_giga = [gold_to_gigaam[k] for k in tok_ids if k in gold_to_gigaam]
        if matched_giga:
            start = min(gigaam[j]["start"] for j in matched_giga)
            end = max(gigaam[j]["end"] for j in matched_giga)
            per_turn.append(
                {
                    "turn": ti + 1,
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "start": start,
                    "end": end,
                    "n_gold_tokens": len(tok_ids),
                    "n_matched": len(matched_giga),
                    "status": "aligned",
                }
            )
        else:
            per_turn.append(
                {
                    "turn": ti + 1,
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "start": None,
                    "end": None,
                    "n_gold_tokens": len(tok_ids),
                    "n_matched": 0,
                    "status": "unaligned",
                }
            )

    # Interpolate unaligned turns between neighbours
    for i, t in enumerate(per_turn):
        if t["status"] == "aligned":
            continue
        prev_end = None
        for j in range(i - 1, -1, -1):
            if per_turn[j]["status"] == "aligned":
                prev_end = per_turn[j]["end"]
                break
        next_start = None
        for j in range(i + 1, len(per_turn)):
            if per_turn[j]["status"] == "aligned":
                next_start = per_turn[j]["start"]
                break
        if prev_end is None and next_start is None:
            t["start"] = 0.0
            t["end"] = TOTAL_DURATION
        elif prev_end is None:
            t["start"] = max(0.0, next_start - 0.5)
            t["end"] = next_start
        elif next_start is None:
            t["start"] = prev_end
            t["end"] = min(TOTAL_DURATION, prev_end + 0.5)
        else:
            midpoint = (prev_end + next_start) / 2.0
            t["start"] = max(prev_end, midpoint - 0.25)
            t["end"] = min(next_start, midpoint + 0.25)
        t["status"] = "interpolated"

    # Sort by start time (gold turns should already be sequential — this is a safety net)
    per_turn.sort(key=lambda x: x["start"])

    # Clip adjacent turn overlaps: if turn_i.end > turn_{i+1}.start, shrink turn_i
    for i in range(len(per_turn) - 1):
        if per_turn[i]["end"] > per_turn[i + 1]["start"]:
            per_turn[i]["end"] = per_turn[i + 1]["start"]

    # Clip negative durations (can happen if alignment inverted)
    fixed = 0
    for t in per_turn:
        if t["end"] < t["start"] + 0.05:  # minimum 50 ms
            t["end"] = t["start"] + 0.05
            fixed += 1
    if fixed:
        print(f"fixed {fixed} zero/negative-duration turns to 50 ms minimum")

    # Clamp to audio duration
    for t in per_turn:
        t["start"] = max(0.0, min(TOTAL_DURATION, t["start"]))
        t["end"] = max(0.0, min(TOTAL_DURATION, t["end"]))

    return per_turn


def _safe_speaker_label(label: str) -> str:
    """RTTM is whitespace-separated; speaker labels with spaces (e.g.
    "Д. Голубев") break the pandas-based RTTM loader in pyannote.database.
    Collapse all runs of whitespace to a single underscore."""
    return "_".join(str(label).split())


def write_rttm(per_turn: list[dict], path: Path, recording_id: str) -> None:
    lines = []
    for t in per_turn:
        start = t["start"]
        dur = max(0.05, t["end"] - t["start"])
        spk = _safe_speaker_label(t["speaker"])
        # RTTM V1: SPEAKER <file> <channel> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
        lines.append(
            f"SPEAKER {recording_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gold = load_gold()
    print(f"loaded {len(gold)} gold turns")
    gigaam = load_gigaam_words()
    print(f"loaded {len(gigaam)} gigaam word timestamps")

    per_turn = align_gold_to_gigaam(gold, gigaam)

    n_aligned = sum(1 for t in per_turn if t["status"] == "aligned")
    n_interp = sum(1 for t in per_turn if t["status"] == "interpolated")
    print(f"turns aligned: {n_aligned}, interpolated: {n_interp}, total: {len(per_turn)}")

    total_speech = sum(t["end"] - t["start"] for t in per_turn)
    print(f"total speech time in reference: {total_speech:.1f}s / {TOTAL_DURATION}s")

    # Per-speaker speaking time
    from collections import Counter
    per_sp = Counter()
    for t in per_turn:
        per_sp[t["speaker"]] += t["end"] - t["start"]
    print("speaker speaking time:")
    for sp, dur in per_sp.most_common():
        print(f"  {sp:16s} {dur:7.1f}s  ({dur / total_speech * 100:5.1f}%)")

    if args.dry_run:
        print("(dry-run, not writing)")
        return

    write_rttm(per_turn, OUT_RTTM, RECORDING_ID)
    OUT_DEBUG.write_text(
        json.dumps(per_turn, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT_RTTM}")
    print(f"wrote {OUT_DEBUG}")


if __name__ == "__main__":
    main()
