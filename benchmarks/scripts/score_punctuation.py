"""Score each punctuation variant against the gold reference.

Inputs:
  benchmarks/gold/court_hearing_129.jsonl — gold turns
  benchmarks/postproc/punctuation/<variant>.txt — variants from runner

For each variant:
  1. Tokenize both sides (lowercase, ё→е, punct detached).
  2. Align on normalised word tokens via difflib.SequenceMatcher.
  3. For each *matched* token pair, record the punctuation suffix attached to
     each side (., ,, ?, !, :, ;). Unmatched tokens are skipped — they indicate
     ASR errors, not punctuation disagreements.
  4. Per-class confusion matrix → precision / recall / F1.

Also computes:
  - sentence_boundary_f1: do both sides agree on where sentences end?
  - A micro-averaged total F1 across all 6 punct classes.

Output:
  benchmarks/reports/punct_scoreboard.json
  benchmarks/reports/punct_scoreboard.html
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_JSONL = ROOT / "benchmarks" / "gold" / "court_hearing_129.jsonl"
VARIANT_DIR = ROOT / "benchmarks" / "postproc" / "punctuation"
OUT_JSON = ROOT / "benchmarks" / "reports" / "punct_scoreboard.json"
OUT_HTML = ROOT / "benchmarks" / "reports" / "punct_scoreboard.html"

# Punctuation classes we care about. Mapped to a single canonical char.
PUNCT_CANON = {
    ".": ".", "!": "!", "?": "?",
    ",": ",", ";": ";", ":": ":",
    "…": ".",  # ellipsis collapses to period for scoring
    "\u2026": ".",
}
SENTENCE_END = {".", "!", "?"}
ALL_PUNCT = [".", ",", "?", "!", ":", ";"]


_PUNCT_CATS = ("P", "S")


def normalise(token: str) -> str:
    token = unicodedata.normalize("NFC", token).lower().replace("\u0451", "\u0435")
    return "".join(ch for ch in token if unicodedata.category(ch)[0] not in _PUNCT_CATS)


def trailing_punct(token: str) -> str:
    """Return the canonical trailing punctuation chars attached to a token.
    Collapse multiple trailing puncts to just the *last* interesting one
    (common case: '!' or '?' wins over '.'). Return '' if none."""
    stripped_end_ws = token.rstrip()
    trail = []
    for ch in reversed(stripped_end_ws):
        if unicodedata.category(ch)[0] in _PUNCT_CATS:
            canon = PUNCT_CANON.get(ch)
            if canon:
                trail.append(canon)
        else:
            break
    # preserve original order
    trail.reverse()
    if not trail:
        return ""
    # Prefer sentence-end over comma if both present
    for p in ("?", "!", "."):
        if p in trail:
            return p
    return trail[-1]  # else take the last canonical punct


def tokenise_with_punct(text: str) -> list[dict]:
    """Return [{norm, punct}] for each whitespace-delimited token. norm is the
    clean word; punct is the canonical trailing punct or ''."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for raw in text.split():
        norm = normalise(raw)
        if not norm:
            continue
        out.append({"norm": norm, "punct": trailing_punct(raw), "orig": raw})
    return out


def load_gold() -> list[dict]:
    tokens = []
    for line in GOLD_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        tokens.extend(tokenise_with_punct(t.get("text", "")))
    return tokens


def load_variant(path: Path) -> list[dict]:
    return tokenise_with_punct(path.read_text(encoding="utf-8"))


# --- Scoring ---

def score_variant(gold: list[dict], hyp: list[dict]) -> dict:
    gn = [t["norm"] for t in gold]
    hn = [t["norm"] for t in hyp]
    sm = SequenceMatcher(a=gn, b=hn, autojunk=False)

    # Per-class confusion: for each punct class p, gold_has / hyp_has.
    # True positive: gold_has == p and hyp_has == p.
    # False positive: hyp_has == p but gold_has != p.
    # False negative: gold_has == p but hyp_has != p.
    tp = Counter(); fp = Counter(); fn = Counter()

    # Sentence boundary: token ends a sentence iff its punct in SENTENCE_END.
    sb_tp = sb_fp = sb_fn = 0

    n_matched = 0
    # Only score on aligned (equal) token pairs — we care about where both
    # sides agreed on the word.
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            gt = gold[i1 + k]
            ht = hyp[j1 + k]
            n_matched += 1
            gp = gt["punct"]
            hp = ht["punct"]
            # Per-class scoring
            for p in ALL_PUNCT:
                g_has = (gp == p)
                h_has = (hp == p)
                if g_has and h_has:
                    tp[p] += 1
                elif h_has and not g_has:
                    fp[p] += 1
                elif g_has and not h_has:
                    fn[p] += 1
            # Sentence-boundary
            g_sb = gp in SENTENCE_END
            h_sb = hp in SENTENCE_END
            if g_sb and h_sb: sb_tp += 1
            elif h_sb and not g_sb: sb_fp += 1
            elif g_sb and not h_sb: sb_fn += 1

    def f1(t, p, n) -> tuple[float, float, float]:
        prec = t / (t + p) if (t + p) else 0.0
        rec = t / (t + n) if (t + n) else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f

    per_class = {}
    for p in ALL_PUNCT:
        prec, rec, f = f1(tp[p], fp[p], fn[p])
        per_class[p] = {
            "tp": tp[p], "fp": fp[p], "fn": fn[p],
            "precision": round(prec, 3),
            "recall":    round(rec, 3),
            "f1":        round(f, 3),
        }
    # Micro average over all 6 classes
    total_tp = sum(tp.values()); total_fp = sum(fp.values()); total_fn = sum(fn.values())
    prec_m, rec_m, f1_m = f1(total_tp, total_fp, total_fn)
    sb_prec, sb_rec, sb_f1 = f1(sb_tp, sb_fp, sb_fn)

    # Sentence count per side
    gold_sents = sum(1 for t in gold if t["punct"] in SENTENCE_END)
    hyp_sents = sum(1 for t in hyp if t["punct"] in SENTENCE_END)

    return {
        "n_matched_pairs": n_matched,
        "gold_tokens": len(gold),
        "hyp_tokens":  len(hyp),
        "gold_sentences": gold_sents,
        "hyp_sentences":  hyp_sents,
        "per_class": per_class,
        "micro": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(prec_m, 3),
            "recall":    round(rec_m, 3),
            "f1":        round(f1_m, 3),
        },
        "sentence_boundary": {
            "tp": sb_tp, "fp": sb_fp, "fn": sb_fn,
            "precision": round(sb_prec, 3),
            "recall":    round(sb_rec, 3),
            "f1":        round(sb_f1, 3),
        },
    }


# --- HTML rendering ---

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def build_html(results: dict) -> str:
    variants = list(results.keys())
    # Scoreboard matrix
    header = "<tr><th>Metric</th>" + "".join(
        f"<th>{html.escape(v)}</th>" for v in variants
    ) + "</tr>"

    def row(label: str, fn):
        cells = "".join(f"<td class='num'>{fn(v)}</td>" for v in variants)
        return f"<tr><td>{label}</td>{cells}</tr>"

    rows = [
        row("Micro-avg F1", lambda v: fmt_pct(results[v]["micro"]["f1"])),
        row("Micro precision", lambda v: fmt_pct(results[v]["micro"]["precision"])),
        row("Micro recall", lambda v: fmt_pct(results[v]["micro"]["recall"])),
        row("Sentence-boundary F1", lambda v: fmt_pct(results[v]["sentence_boundary"]["f1"])),
        row("Sentence count (gold / hyp)", lambda v: f"{results[v]['gold_sentences']} / {results[v]['hyp_sentences']}"),
        row("Matched token pairs", lambda v: f"{results[v]['n_matched_pairs']:,}"),
    ]
    for p in ALL_PUNCT:
        rows.append(
            row(f"F1 '{p}'", lambda v, _p=p: fmt_pct(results[v]["per_class"][_p]["f1"]))
        )
        rows.append(
            row(f" &nbsp;&nbsp; P / R", lambda v, _p=p: f"{fmt_pct(results[v]['per_class'][_p]['precision'])} / {fmt_pct(results[v]['per_class'][_p]['recall'])}")
        )
        rows.append(
            row(f" &nbsp;&nbsp; tp / fp / fn", lambda v, _p=p: f"{results[v]['per_class'][_p]['tp']} / {results[v]['per_class'][_p]['fp']} / {results[v]['per_class'][_p]['fn']}")
        )

    css = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 16px 24px;
       color: #222; background: #fafafa; }
h1 { color: #B2001F; font-family: Georgia, serif; margin: 0 0 4px 0; font-size: 22px; }
h2 { font-family: Georgia, serif; margin: 22px 0 8px 0; color: #1f1f1f; font-size: 16px; }
.meta { color: #555; font-size: 13px; margin-bottom: 14px; line-height: 1.5; }
table { border-collapse: collapse; background: #fff; font-size: 13px; }
table th, table td { border: 1px solid #ddd; padding: 6px 14px; text-align: left; }
table thead th { background: #1f1f1f; color: #fff; }
table td.num { text-align: right; font-variant-numeric: tabular-nums;
                font-family: Consolas, monospace; }
table tbody tr:nth-child(even) td { background: #f7f7f7; }
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Punctuation scoreboard — Court Hearing 129</title>
<style>{css}</style>
</head><body>
<h1>Punctuation scoreboard — Court Hearing 129</h1>
<div class="meta">
Three punctuation sources scored against gold. For each variant the text was
tokenised, aligned to gold via difflib, and per-class confusion was computed on
<b>matched word pairs only</b> (mismatched words are skipped — those are ASR
errors, not punctuation errors).<br>
Sentence-boundary F1 measures agreement on where <code>.!?</code> occur.
</div>
<table><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>
</body></html>
"""


def main() -> None:
    if not GOLD_JSONL.exists():
        raise SystemExit(f"missing gold: {GOLD_JSONL}")
    gold = load_gold()
    print(f"gold tokens: {len(gold)}")

    results = {}
    for path in sorted(VARIANT_DIR.glob("*.txt")):
        if path.name.startswith("_"):
            continue
        variant = path.stem
        print(f"\n=== {variant} ===")
        hyp = load_variant(path)
        print(f"  hyp tokens: {len(hyp)}")
        r = score_variant(gold, hyp)
        print(f"  micro F1: {r['micro']['f1']}  (P {r['micro']['precision']} / R {r['micro']['recall']})")
        print(f"  sentence-boundary F1: {r['sentence_boundary']['f1']}")
        for p in ALL_PUNCT:
            pc = r["per_class"][p]
            print(f"    '{p}'  tp={pc['tp']:4d} fp={pc['fp']:4d} fn={pc['fn']:4d}  "
                  f"P={pc['precision']}  R={pc['recall']}  F1={pc['f1']}")
        results[variant] = r

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_HTML.write_text(build_html(results), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
