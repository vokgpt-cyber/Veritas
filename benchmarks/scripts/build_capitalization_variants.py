"""Build capitalization variants from GigaAM native ASR output.

Starts from benchmarks/postproc/punctuation/gigaam_native.txt (which already
has GigaAM's native casing) and produces parallel variants so we can score
which rule gives the best agreement with gold casing.

Variants produced:

  as_is.txt
    Straight copy of gigaam_native.txt. Baseline.

  lower_then_sentence_caps.txt
    Lowercase everything, then re-apply a simple sentence-start rule
    (capitalize first word of text + first word after .!?). This is the
    pessimistic "what if GigaAM produced no casing at all" case — it shows
    what our rule-based recovery alone can do.

  sentence_caps_overlay.txt
    Keep GigaAM's native casing AND force-capitalize any first-word-after-.!?
    that is still lowercase. This is the in-pipeline rule the postprocessor
    should apply: non-destructive, only fills missed sentence starts.

  oracle_proper_nouns.txt
    Cheating-but-informative ceiling. For every word-type that appears
    majority-capitalized in gold, capitalize it everywhere in the ASR.
    This tells us how much headroom there is above rule-based
    sentence casing — i.e. the ceiling for a proper-noun NER pass.

Inputs:
  benchmarks/postproc/punctuation/gigaam_native.txt
  benchmarks/gold/court_hearing_129.jsonl  (for oracle proper-noun list)

Outputs:
  benchmarks/postproc/capitalization/<variant>.txt
  benchmarks/postproc/capitalization/manifest.json  (variant descriptions)
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "benchmarks" / "postproc" / "punctuation" / "gigaam_native.txt"
GOLD = ROOT / "benchmarks" / "gold" / "court_hearing_129.jsonl"
OUT_DIR = ROOT / "benchmarks" / "postproc" / "capitalization"


_PUNCT_CATS = ("P", "S")


def normalise_word(token: str) -> str:
    """Lowercase + strip punctuation. Used as the dictionary key for oracle."""
    token = unicodedata.normalize("NFC", token).lower().replace("\u0451", "\u0435")
    return "".join(ch for ch in token if unicodedata.category(ch)[0] not in _PUNCT_CATS)


def split_preserve_punct(text: str) -> list[str]:
    """Split on whitespace, keep tokens verbatim (punct attached)."""
    return text.split()


def rebuild(tokens: list[str]) -> str:
    """Re-join tokens with single space."""
    return " ".join(tokens)


def is_alpha_letter(ch: str) -> bool:
    return ch.isalpha()


def first_alpha_index(token: str) -> int:
    """Index of the first alphabetic character in a token, or -1."""
    for i, ch in enumerate(token):
        if is_alpha_letter(ch):
            return i
    return -1


def capitalize_first_letter(token: str) -> str:
    """Capitalize the first alphabetic character in a token (keep rest)."""
    i = first_alpha_index(token)
    if i < 0:
        return token
    return token[:i] + token[i].upper() + token[i + 1 :]


def lowercase_first_letter(token: str) -> str:
    i = first_alpha_index(token)
    if i < 0:
        return token
    return token[:i] + token[i].lower() + token[i + 1 :]


# --- Variant builders ---


def variant_as_is(text: str) -> str:
    return text


def variant_lower_then_sentence_caps(text: str) -> str:
    """Full-lowercase baseline with sentence-start caps re-applied.

    Shows what rule-based casing alone (no native) can recover.
    """
    low = text.lower()
    tokens = split_preserve_punct(low)
    if not tokens:
        return low
    # Force capitalize the first alphabetic token
    tokens[0] = capitalize_first_letter(tokens[0])
    # Capitalize any token whose previous token ended in .!?
    SENT_END_CHARS = set(".!?…")
    for i in range(1, len(tokens)):
        prev = tokens[i - 1].rstrip()
        # Strip quotes/brackets off the end for robustness
        while prev and prev[-1] in "\"»')]":
            prev = prev[:-1]
        if prev and prev[-1] in SENT_END_CHARS:
            tokens[i] = capitalize_first_letter(tokens[i])
    return rebuild(tokens)


def variant_sentence_caps_overlay(text: str) -> str:
    """Keep GigaAM casing, additionally cap any post-sentence-end token still
    starting with a lowercase letter. Non-destructive overlay."""
    tokens = split_preserve_punct(text)
    if not tokens:
        return text
    SENT_END_CHARS = set(".!?…")
    # First token: ensure capitalized
    i0 = first_alpha_index(tokens[0])
    if i0 >= 0 and tokens[0][i0].islower():
        tokens[0] = capitalize_first_letter(tokens[0])
    for i in range(1, len(tokens)):
        prev = tokens[i - 1].rstrip()
        while prev and prev[-1] in "\"»')]":
            prev = prev[:-1]
        if prev and prev[-1] in SENT_END_CHARS:
            idx = first_alpha_index(tokens[i])
            if idx >= 0 and tokens[i][idx].islower():
                tokens[i] = capitalize_first_letter(tokens[i])
    return rebuild(tokens)


def build_proper_noun_set(gold_path: Path, min_count: int = 2, cap_share: float = 0.7) -> set[str]:
    """From gold, find word lemmas that appear majority-capitalized.

    A word 'key' = normalised (lowercase, no punct). For each key, tally how
    many times it appeared capitalized vs lowercased in gold. If it appears
    at least `min_count` times AND at least `cap_share` of occurrences are
    capitalized, treat it as a proper noun.
    """
    cap_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()

    for line in gold_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        for tok in split_preserve_punct(t.get("text", "")):
            key = normalise_word(tok)
            if not key:
                continue
            idx = first_alpha_index(tok)
            if idx < 0:
                continue
            first = tok[idx]
            if first.isupper():
                cap_counts[key] += 1
            else:
                low_counts[key] += 1

    proper: set[str] = set()
    for key in set(cap_counts) | set(low_counts):
        n_up = cap_counts[key]
        n_lo = low_counts[key]
        total = n_up + n_lo
        if total < min_count:
            continue
        # Skip single-letter keys — likely initials / noise
        if len(key) <= 1:
            continue
        if n_up / total >= cap_share and n_up >= 1:
            # Reject very common stopwords accidentally tagged (they'd tend
            # to be cap_share < 0.7 but sanity-check anyway).
            proper.add(key)
    return proper


def variant_oracle_proper_nouns(text: str, proper_set: set[str]) -> str:
    """Overlay proper-noun casing from gold. Starts from sentence_caps_overlay."""
    text = variant_sentence_caps_overlay(text)
    tokens = split_preserve_punct(text)
    out: list[str] = []
    for tok in tokens:
        key = normalise_word(tok)
        if key in proper_set:
            tok = capitalize_first_letter(tok)
        out.append(tok)
    return rebuild(out)


# --- Main ---


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source: {SRC}")
    if not GOLD.exists():
        raise SystemExit(f"missing gold: {GOLD}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = SRC.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"empty source: {SRC}")

    proper = build_proper_noun_set(GOLD)
    print(f"oracle proper-noun set: {len(proper)} lemmas")

    variants = {
        "as_is.txt": variant_as_is(text),
        "lower_then_sentence_caps.txt": variant_lower_then_sentence_caps(text),
        "sentence_caps_overlay.txt": variant_sentence_caps_overlay(text),
        "oracle_proper_nouns.txt": variant_oracle_proper_nouns(text, proper),
    }

    for fname, content in variants.items():
        path = OUT_DIR / fname
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.name}  ({len(content):,} chars)")

    manifest = {
        "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "gold": str(GOLD.relative_to(ROOT)).replace("\\", "/"),
        "oracle_proper_noun_count": len(proper),
        "oracle_proper_noun_sample": sorted(list(proper))[:40],
        "variants": {
            "as_is.txt": "Verbatim gigaam_native.txt — native casing baseline.",
            "lower_then_sentence_caps.txt":
                "Text lowercased, then sentence-start rule re-applied "
                "(first token + first token after .!?). Pessimistic case.",
            "sentence_caps_overlay.txt":
                "Keep GigaAM casing, overlay sentence-start rule only on "
                "lowercase offenders. Non-destructive. This is the rule the "
                "postprocessor should apply.",
            "oracle_proper_nouns.txt":
                "sentence_caps_overlay + oracle proper-noun dictionary mined "
                "from gold. Ceiling for a perfect NER pass.",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {(OUT_DIR / 'manifest.json').name}")


if __name__ == "__main__":
    main()
