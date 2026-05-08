"""Compute Russian-friendly WER / CER between an ASR plain.txt and a gold plain.txt.

Usage:
    python benchmarks/scripts/score_wer.py \
        --ref benchmarks/gold/court_hearing_129_plain.txt \
        --hyp benchmarks/asr/gigaam_v3/court_hearing_129_plain.txt \
        --out benchmarks/metrics/asr_gigaam_v3_court_hearing_129.json \
        --tag gigaam_v3

Normalisation (applied to BOTH hyp and ref before scoring):
    - NFC unicode normalisation
    - Lowercase
    - ё -> е (Russian orthographic quirk, not a substantive error)
    - Strip punctuation (Unicode categories P and S)
    - Collapse whitespace

Metrics reported:
    wer, cer, n_ref_words, n_hyp_words, substitutions, deletions, insertions, hits

Dependencies:
    - jiwer >= 4.0 (uses jiwer.process_words; older API `compute_measures` was removed)
    - rapidfuzz (for CER via character-level Levenshtein)
    Install: pip install jiwer rapidfuzz
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace("\u0451", "\u0435")  # ё -> е
    text = text.replace("\u0401", "\u0415")  # Ё -> Е (redundant after lower but safe)
    out_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    text = "".join(out_chars)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    return text.split() if text else []


def score(ref_text: str, hyp_text: str, *, skip_cer: bool = False) -> dict:
    ref_norm = normalise(ref_text)
    hyp_norm = normalise(hyp_text)
    ref_words = tokenise(ref_norm)
    hyp_words = tokenise(hyp_norm)
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)

    import jiwer  # type: ignore

    wo = jiwer.process_words(ref_norm, hyp_norm)
    sub = int(wo.substitutions)
    dele = int(wo.deletions)
    ins = int(wo.insertions)
    hits = int(wo.hits)
    wer = float(wo.wer)

    cer: float | None = None
    if not skip_cer:
        try:
            from rapidfuzz.distance import Levenshtein as RFL  # type: ignore

            ref_chars = ref_norm.replace(" ", "")
            hyp_chars = hyp_norm.replace(" ", "")
            if ref_chars:
                cer = RFL.distance(ref_chars, hyp_chars) / len(ref_chars)
        except ImportError:
            try:
                cer = float(jiwer.cer(ref_norm, hyp_norm))
            except Exception:
                cer = None

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4) if cer is not None else None,
        "n_ref_words": n_ref,
        "n_hyp_words": n_hyp,
        "hits": hits,
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="gold plain.txt")
    ap.add_argument("--hyp", required=True, type=Path, help="ASR plain.txt")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON metrics output")
    ap.add_argument("--tag", default="", help="optional label echoed into JSON")
    ap.add_argument("--skip-cer", action="store_true", help="skip CER")
    args = ap.parse_args()

    ref_text = args.ref.read_text(encoding="utf-8")
    hyp_text = args.hyp.read_text(encoding="utf-8")
    result = score(ref_text, hyp_text, skip_cer=args.skip_cer)
    result["ref_file"] = str(args.ref)
    result["hyp_file"] = str(args.hyp)
    if args.tag:
        result["tag"] = args.tag

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
