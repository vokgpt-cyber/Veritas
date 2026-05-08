"""Score each capitalization variant against the gold reference.

Inputs:
  benchmarks/gold/court_hearing_129.jsonl
  benchmarks/postproc/capitalization/<variant>.txt

Method:
  1. Tokenise both sides (whitespace split, keep originals).
  2. Normalise each token for alignment: lowercase, ё→е, strip punct/symbols.
  3. Align on normalised tokens via difflib.SequenceMatcher.
  4. For each *matched* (tag == "equal") pair, compare the *first alphabetic
     character* of each side's original token. Classify:
        correct         — both upper or both lower
        missed_capital  — gold upper, hyp lower  (we should have capitalized)
        over_capital    — gold lower, hyp upper  (we over-capitalized)
     Unmatched tokens are skipped — those are ASR word errors, not cap errors.

  5. Report totals, accuracy, per-direction F1 for the 'capital-letter' class.
  6. List top-20 offending word lemmas per direction per variant.

Also writes benchmarks/reports/cap_scoreboard.json/html.
"""
from __future__ import annotations

import html
import json
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_JSONL = ROOT / "benchmarks" / "gold" / "court_hearing_129.jsonl"
VARIANT_DIR = ROOT / "benchmarks" / "postproc" / "capitalization"
OUT_JSON = ROOT / "benchmarks" / "reports" / "cap_scoreboard.json"
OUT_HTML = ROOT / "benchmarks" / "reports" / "cap_scoreboard.html"

_PUNCT_CATS = ("P", "S")


def normalise(token: str) -> str:
    token = unicodedata.normalize("NFC", token).lower().replace("\u0451", "\u0435")
    return "".join(ch for ch in token if unicodedata.category(ch)[0] not in _PUNCT_CATS)


def first_alpha_case(token: str) -> str | None:
    """Return 'U' if first alpha char is uppercase, 'L' if lowercase, None if
    the token has no alphabetic character."""
    for ch in token:
        if ch.isalpha():
            return "U" if ch.isupper() else "L"
    return None


def tokenise(text: str) -> list[dict]:
    """Split on whitespace, keep the original and record normalised form +
    first-letter case."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for raw in text.split():
        norm = normalise(raw)
        if not norm:
            continue
        out.append({"norm": norm, "case": first_alpha_case(raw), "orig": raw})
    return out


def load_gold() -> list[dict]:
    tokens = []
    for line in GOLD_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        tokens.extend(tokenise(t.get("text", "")))
    return tokens


def load_variant(path: Path) -> list[dict]:
    return tokenise(path.read_text(encoding="utf-8"))


# --- Scoring ---

def score_variant(gold: list[dict], hyp: list[dict]) -> dict:
    gn = [t["norm"] for t in gold]
    hn = [t["norm"] for t in hyp]
    sm = SequenceMatcher(a=gn, b=hn, autojunk=False)

    n_matched = 0
    correct = 0
    missed_cap = 0  # gold U, hyp L
    over_cap = 0    # gold L, hyp U
    noalpha = 0     # no alphabetic char either side — skipped from pct

    # For "capitalized-letter" as a class: we treat it as binary per token.
    # precision / recall / f1 of predicting "this token starts with a capital."
    tp = fp = fn = tn = 0

    missed_by_lemma: Counter[str] = Counter()
    over_by_lemma: Counter[str] = Counter()
    # Also record example origs for offenders.
    missed_examples: dict[str, str] = {}
    over_examples: dict[str, str] = {}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            g = gold[i1 + k]
            h = hyp[j1 + k]
            n_matched += 1
            gc = g["case"]
            hc = h["case"]
            if gc is None or hc is None:
                noalpha += 1
                continue
            if gc == hc:
                correct += 1
            elif gc == "U" and hc == "L":
                missed_cap += 1
                missed_by_lemma[g["norm"]] += 1
                missed_examples.setdefault(g["norm"], f"gold={g['orig']!r} hyp={h['orig']!r}")
            elif gc == "L" and hc == "U":
                over_cap += 1
                over_by_lemma[g["norm"]] += 1
                over_examples.setdefault(g["norm"], f"gold={g['orig']!r} hyp={h['orig']!r}")
            # Per-class confusion for "is capitalized"
            if gc == "U" and hc == "U":
                tp += 1
            elif gc == "L" and hc == "U":
                fp += 1
            elif gc == "U" and hc == "L":
                fn += 1
            else:
                tn += 1

    denom = max(1, n_matched - noalpha)
    accuracy = correct / denom

    def f1(t: int, p: int, n: int) -> tuple[float, float, float]:
        prec = t / (t + p) if (t + p) else 0.0
        rec = t / (t + n) if (t + n) else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f

    prec, rec, f = f1(tp, fp, fn)

    # Top-20 offenders per direction
    top_missed = [
        {"lemma": k, "count": c, "example": missed_examples.get(k, "")}
        for k, c in missed_by_lemma.most_common(20)
    ]
    top_over = [
        {"lemma": k, "count": c, "example": over_examples.get(k, "")}
        for k, c in over_by_lemma.most_common(20)
    ]

    return {
        "gold_tokens": len(gold),
        "hyp_tokens": len(hyp),
        "n_matched_pairs": n_matched,
        "no_alpha_pairs": noalpha,
        "correct": correct,
        "missed_capital": missed_cap,
        "over_capital": over_cap,
        "accuracy": round(accuracy, 4),
        "capital_class": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f, 4),
        },
        "top_missed": top_missed,
        "top_over": top_over,
    }


# --- HTML rendering ---

def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def build_html(results: dict) -> str:
    variants = list(results.keys())
    header = "<tr><th>Metric</th>" + "".join(
        f"<th>{html.escape(v)}</th>" for v in variants
    ) + "</tr>"

    def row(label: str, fn):
        cells = "".join(f"<td class='num'>{fn(v)}</td>" for v in variants)
        return f"<tr><td>{label}</td>{cells}</tr>"

    rows = [
        row("Accuracy (matched pairs)", lambda v: fmt_pct(results[v]["accuracy"])),
        row("Capital-class F1", lambda v: fmt_pct(results[v]["capital_class"]["f1"])),
        row(" &nbsp;&nbsp;P / R",
            lambda v: f"{fmt_pct(results[v]['capital_class']['precision'])} / {fmt_pct(results[v]['capital_class']['recall'])}"),
        row("Missed capital (gold U, hyp L)", lambda v: f"{results[v]['missed_capital']:,}"),
        row("Over capital (gold L, hyp U)", lambda v: f"{results[v]['over_capital']:,}"),
        row("Cap mismatch total", lambda v: f"{results[v]['missed_capital'] + results[v]['over_capital']:,}"),
        row("Matched token pairs", lambda v: f"{results[v]['n_matched_pairs']:,}"),
        row("Gold tokens / Hyp tokens",
            lambda v: f"{results[v]['gold_tokens']:,} / {results[v]['hyp_tokens']:,}"),
    ]

    # Top offenders tables, one per variant
    offenders_blocks = []
    for v in variants:
        r = results[v]
        missed_rows = "".join(
            f"<tr><td>{html.escape(x['lemma'])}</td><td class='num'>{x['count']}</td>"
            f"<td>{html.escape(x['example'])}</td></tr>"
            for x in r["top_missed"]
        ) or "<tr><td colspan='3'><em>(none)</em></td></tr>"
        over_rows = "".join(
            f"<tr><td>{html.escape(x['lemma'])}</td><td class='num'>{x['count']}</td>"
            f"<td>{html.escape(x['example'])}</td></tr>"
            for x in r["top_over"]
        ) or "<tr><td colspan='3'><em>(none)</em></td></tr>"
        offenders_blocks.append(
            f"<h2>{html.escape(v)}</h2>"
            f"<div class='two-col'>"
            f"<div><h3>Top missed capitals</h3>"
            f"<table><thead><tr><th>Lemma</th><th>Count</th><th>Example</th></tr></thead>"
            f"<tbody>{missed_rows}</tbody></table></div>"
            f"<div><h3>Top over-capitals</h3>"
            f"<table><thead><tr><th>Lemma</th><th>Count</th><th>Example</th></tr></thead>"
            f"<tbody>{over_rows}</tbody></table></div>"
            f"</div>"
        )

    css = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 16px 24px;
       color: #222; background: #fafafa; }
h1 { color: #B2001F; font-family: Georgia, serif; margin: 0 0 4px 0; font-size: 22px; }
h2 { font-family: Georgia, serif; margin: 22px 0 8px 0; color: #1f1f1f; font-size: 16px; }
h3 { font-family: Georgia, serif; margin: 14px 0 6px 0; color: #1f1f1f; font-size: 14px; }
.meta { color: #555; font-size: 13px; margin-bottom: 14px; line-height: 1.5; }
table { border-collapse: collapse; background: #fff; font-size: 13px; margin-bottom: 8px; }
table th, table td { border: 1px solid #ddd; padding: 6px 12px; text-align: left; }
table thead th { background: #1f1f1f; color: #fff; }
table td.num { text-align: right; font-variant-numeric: tabular-nums;
                font-family: Consolas, monospace; }
table tbody tr:nth-child(even) td { background: #f7f7f7; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Capitalization scoreboard — Court Hearing 129</title>
<style>{css}</style>
</head><body>
<h1>Capitalization scoreboard — Court Hearing 129</h1>
<div class="meta">
Each variant compared against gold casing on <b>matched word pairs only</b>
(mismatched words are ASR errors, not cap errors — skipped).<br>
Capital-class F1 = treating "token starts with an uppercase letter" as a
binary classification task. Higher = better agreement with gold casing.
</div>
<table><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>
{"".join(offenders_blocks)}
</body></html>
"""


def main() -> None:
    if not GOLD_JSONL.exists():
        raise SystemExit(f"missing gold: {GOLD_JSONL}")
    gold = load_gold()
    print(f"gold tokens: {len(gold)}")

    results: dict[str, dict] = {}
    for path in sorted(VARIANT_DIR.glob("*.txt")):
        if path.name.startswith("_"):
            continue
        variant = path.stem
        print(f"\n=== {variant} ===")
        hyp = load_variant(path)
        print(f"  hyp tokens: {len(hyp)}")
        r = score_variant(gold, hyp)
        print(f"  accuracy:              {r['accuracy']:.4f}")
        print(f"  capital-class F1:      {r['capital_class']['f1']:.4f}  "
              f"(P {r['capital_class']['precision']:.4f} / R {r['capital_class']['recall']:.4f})")
        print(f"  missed capital:        {r['missed_capital']}")
        print(f"  over capital:          {r['over_capital']}")
        print(f"  total cap mismatches:  {r['missed_capital'] + r['over_capital']}")
        results[variant] = r

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_HTML.write_text(build_html(results), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
