"""Produce three punctuation variants of the GigaAM transcript for court_hearing_129.

Why: GigaAM emits punctuation natively, but we don't know whether a dedicated
punctuation model would do better. This script strips punctuation from GigaAM's
output and re-attaches it via two alternative models, so we can A/B them against
the gold reference.

Variants produced:
  gigaam_native.txt  — raw GigaAM text, punct preserved as-is (baseline).
  dmp_repunct.txt    — stripped → deepmultilingualpunctuation (BERT, multilingual).
  silero_repunct.txt — stripped → silero ru_punct (RU-specific).

Output:
  benchmarks/postproc/punctuation/<variant>.txt
  benchmarks/postproc/punctuation/manifest.json

Usage:
  python benchmarks/scripts/run_punctuation_variants.py
  python benchmarks/scripts/run_punctuation_variants.py --only gigaam_native
  python benchmarks/scripts/run_punctuation_variants.py --only dmp

Run on the VERITAS machine (RTX 3090). DMP is ~300 MB BERT, runs on CPU.
Silero ru_punct is ~65 MB, CPU only. Neither uses GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GIGAAM_SEG = ROOT / "benchmarks" / "asr" / "gigaam_v3" / "court_hearing_129_segments.json"
OUT_DIR = ROOT / "benchmarks" / "postproc" / "punctuation"


_PUNCT_CATS = ("P", "S")


def strip_punct(text: str) -> str:
    """Lowercase the text, strip all punctuation and symbols; keep whitespace.
    Purpose: give DMP / Silero a clean input so they don't see existing punct
    and merely copy it."""
    text = unicodedata.normalize("NFC", text).lower().replace("\u0451", "\u0435")
    out = []
    for ch in text:
        if unicodedata.category(ch)[0] in _PUNCT_CATS:
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def gigaam_concat_text() -> tuple[str, str]:
    """Return (native_text, stripped_text). Segments are joined by a space in
    segment order."""
    segs = json.loads(GIGAAM_SEG.read_text(encoding="utf-8"))
    segs.sort(key=lambda s: float(s.get("start", 0)))
    native = " ".join(s.get("text", "").strip() for s in segs if s.get("text"))
    native = re.sub(r"\s+", " ", native).strip()
    return native, strip_punct(native)


# --- Variant runners ---

def variant_gigaam_native(native: str) -> str:
    return native


def variant_dmp(stripped: str) -> str:
    """deepmultilingualpunctuation by oliverguhr — BERT-based.

    Model: oliverguhr/fullstop-punctuation-multilang-large. Adds periods,
    commas, question marks for English/German/French/Italian/Russian. On Russian
    texts it's workable but not native-language-specific.
    """
    try:
        from deepmultilingualpunctuation import PunctuationModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"deepmultilingualpunctuation not installed: {e}")
    t0 = time.perf_counter()
    model = PunctuationModel()
    print(f"  dmp: loaded in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    restored = model.restore_punctuation(stripped)
    print(f"  dmp: restored in {time.perf_counter() - t0:.1f}s")
    return restored


def variant_silero(stripped: str) -> str:
    """Silero ru_punct — Russian-specific punctuation + true-casing restorer.

    Silero models expose a one-liner via torch.hub. The te_model wraps a small
    transformer trained on Russian news/wiki + conversational data.
    """
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"torch not installed: {e}")
    t0 = time.perf_counter()
    # Silero ships a hub entry; cache is offline-friendly on second run.
    try:
        model, example_texts, languages, punct, apply_te = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_te",
            trust_repo=True,
        )
    except Exception as e:
        raise RuntimeError(f"silero_te load failed: {e}")
    print(f"  silero: loaded in {time.perf_counter() - t0:.1f}s")
    # apply_te(text, lan='ru') restores punctuation AND casing.
    t0 = time.perf_counter()
    # apply_te has a hard internal sentence/chunk limit; chunk manually to be safe.
    chunks = _chunk_words(stripped, max_words=200)
    pieces = [apply_te(c, lan="ru") for c in chunks]
    restored = " ".join(pieces)
    print(f"  silero: restored in {time.perf_counter() - t0:.1f}s"
          f" ({len(chunks)} chunks)")
    return restored


def _chunk_words(text: str, max_words: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


VARIANTS = {
    "gigaam_native": variant_gigaam_native,
    "dmp":           variant_dmp,
    "silero":        variant_silero,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(VARIANTS.keys()),
                    help="Run only the listed variants.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading GigaAM segments from {GIGAAM_SEG.name} ...")
    native, stripped = gigaam_concat_text()
    print(f"  native: {len(native)} chars / {len(native.split())} words")
    print(f"  stripped: {len(stripped)} chars / {len(stripped.split())} words")

    # Sanity: stripped should have same word count as native tokens
    (OUT_DIR / "_input_stripped.txt").write_text(stripped, encoding="utf-8")

    manifest = {
        "source": str(GIGAAM_SEG.relative_to(ROOT)),
        "n_words_stripped": len(stripped.split()),
        "variants": {},
    }

    selected = set(args.only) if args.only else set(VARIANTS.keys())

    for name, runner in VARIANTS.items():
        if name not in selected:
            print(f"[skip] {name}")
            continue
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        try:
            if name == "gigaam_native":
                text = runner(native)
            else:
                text = runner(stripped)
        except Exception as e:
            print(f"  FAILED: {e}")
            manifest["variants"][name] = {"status": "error", "error": str(e)}
            continue
        elapsed = time.perf_counter() - t0
        out = OUT_DIR / f"{name}.txt"
        out.write_text(text, encoding="utf-8")
        manifest["variants"][name] = {
            "status": "ok",
            "path": str(out.relative_to(ROOT)),
            "elapsed_s": round(elapsed, 2),
            "n_chars": len(text),
            "n_words": len(text.split()),
        }
        print(f"  wrote {out}  ({len(text)} chars, {elapsed:.1f}s)")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
