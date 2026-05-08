"""Parse the court-hearing gold transcript from a two-column .docx table.

Input:
    data/test/Court hearing TRUTH 129_1.docx
        Single table. Column 0 = speaker label ("Суд:", "Д. Голубев:", ...).
        Column 1 = Russian utterance text.

Outputs (written to benchmarks/gold/):
    court_hearing_129.jsonl        one JSON object per row: {turn, speaker, text}
    court_hearing_129_plain.txt    concatenated text only, for WER scoring
    court_hearing_129_speakers.txt speaker-tagged plaintext, for visual diff
    court_hearing_129_metadata.json { num_turns, speakers, char_counts, word_count }

We strip trailing colons from speaker labels and collapse whitespace. We keep
the original Russian text byte-for-byte; downstream WER normalisation (lower-
casing, punctuation stripping) happens in the eval script, not here.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from docx import Document

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCX_PATH = REPO_ROOT / "data" / "test" / "Court hearing TRUTH 129_1.docx"
OUT_DIR = REPO_ROOT / "benchmarks" / "gold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "court_hearing_129"

WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalise_cell(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def strip_speaker(label: str) -> str:
    label = normalise_cell(label)
    # Drop trailing punctuation commonly on speaker labels.
    return label.rstrip(":\u2013\u2014.- ")


def main() -> None:
    if not DOCX_PATH.exists():
        raise SystemExit(f"Gold transcript not found: {DOCX_PATH}")

    doc = Document(str(DOCX_PATH))
    if not doc.tables:
        raise SystemExit("No tables found in gold docx.")

    table = doc.tables[0]
    turns: list[dict] = []
    last_speaker: str | None = None

    for row in table.rows:
        cells = [normalise_cell(c.text) for c in row.cells]
        if not any(cells):
            continue
        # Expect [speaker, text]. Handle merged or shifted rows gracefully.
        if len(cells) == 1:
            speaker, text = "", cells[0]
        else:
            speaker, text = cells[0], cells[1]

        speaker_clean = strip_speaker(speaker)
        if not speaker_clean and last_speaker:
            speaker_clean = last_speaker  # continuation row
        if speaker_clean:
            last_speaker = speaker_clean
        if not text:
            continue

        turns.append(
            {
                "turn": len(turns) + 1,
                "speaker": speaker_clean or "UNKNOWN",
                "text": text,
            }
        )

    # JSONL (one turn per line, UTF-8, no ensure_ascii).
    jsonl_path = OUT_DIR / f"{BASE}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for turn in turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    # Plain text for WER scoring (no speaker labels, one line per utterance).
    plain_path = OUT_DIR / f"{BASE}_plain.txt"
    with plain_path.open("w", encoding="utf-8", newline="\n") as f:
        for turn in turns:
            f.write(turn["text"] + "\n")

    # Speaker-tagged plain text for visual comparison (Markdown-ish).
    tagged_path = OUT_DIR / f"{BASE}_speakers.txt"
    with tagged_path.open("w", encoding="utf-8", newline="\n") as f:
        for turn in turns:
            f.write(f"[{turn['speaker']}] {turn['text']}\n")

    # Metadata summary.
    speaker_counts = Counter(t["speaker"] for t in turns)
    char_counts = {s: sum(len(t["text"]) for t in turns if t["speaker"] == s) for s in speaker_counts}
    word_count = sum(len(t["text"].split()) for t in turns)
    meta = {
        "source_docx": str(DOCX_PATH.name),
        "num_turns": len(turns),
        "num_speakers": len(speaker_counts),
        "speakers": speaker_counts.most_common(),
        "char_counts_by_speaker": char_counts,
        "total_chars": sum(len(t["text"]) for t in turns),
        "total_words": word_count,
    }
    meta_path = OUT_DIR / f"{BASE}_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(turns)} turns to {jsonl_path.name}")
    print(f"Plain text:          {plain_path.name}  ({sum(len(t['text']) for t in turns)} chars)")
    print(f"Speaker-tagged:      {tagged_path.name}")
    print(f"Metadata:            {meta_path.name}")
    print()
    print("Speaker distribution:")
    for speaker, count in speaker_counts.most_common():
        chars = char_counts[speaker]
        print(f"  {speaker:<15} {count:>4} turns   {chars:>6} chars")


if __name__ == "__main__":
    main()
