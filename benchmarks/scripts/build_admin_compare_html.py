#!/usr/bin/env python3
"""Build a three-column side-by-side HTML for the Админ 13-04 meeting.

Columns (left → right):
  A. IT team's output        — reference; real speaker names.
                               IT audio starts ~7.5 min before ours (we
                               trimmed pre-meeting chatter) — timestamps
                               are auto-shifted to line up with ours.
  B. Our old pipeline        — pipeline output AS IT WAS when archived.
                               For the Apr-20 Админ archive that's the
                               pre-Phase-A+ aligner + pre-Phase-A+
                               postprocessor. Source: archive's
                               aligned.json. Override with
                               --old-variant repolished to instead load
                               aligned_v2_polished.json (same archived
                               ASR/diar bytes re-run through current
                               postprocessor) if you want to isolate
                               postprocessor changes specifically.
  C. Our fresh E2E run       — fresh ASR + fresh diarization + current
                               postprocessor, produced by run_admin_e2e.py.

Under the default "original" old-variant, B vs C shows the end-to-end
improvement our Phase-A+ work delivered; A is the independent reference.
Switch old-variant to "repolished" to isolate postprocessor-only gains
on identical ASR/diar bytes.

Alignment strategy: the IT team's transcript has the finest time
granularity and is our "anchor" timeline. For every IT turn we scan
both our old and our fresh transcripts for overlapping turns and render
them side-by-side on the same row. Reader can scan horizontally to see
what each stack produced for the same moment in the meeting.

Target-phrase callouts (pinned at the top of the page) highlight:
  1. «Саш, не говори, пожалуйста, что это невозможно» (the isolated
     short-interjection case — did each stack attribute it correctly?)
  2. The 12:30–14:06 finance exchange (Алена Мартынова / Роман Жавнер)
     — this is where the user flagged ASR + turn-boundary gaps.

For our columns we show `speaker_id` (or `speaker_name` if renamed) and
a color-coded `attribution_confidence` badge (green ≥0.8, amber ≥0.6,
red <0.6) when available.

Usage:
    python benchmarks/scripts/build_admin_compare_html.py \\
        --it-team path/to/DiarizerWhisper.txt \\
        --old     path/to/old_archive_dir/ \\
        --fresh   path/to/fresh_e2e_dir/ \\
        --out     benchmarks/reports/admin_13_04_compare.html

If --old or --fresh are omitted the corresponding column shows a
"missing" placeholder — useful for running the builder before the
fresh E2E has completed.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Target passages to pin at the top of the page.
TARGET_PHRASES = [
    ("не говори, пожалуйста", "short-interjection attribution test"),
    ("это невозможно", "short-interjection attribution test (alt match)"),
    ("годовой договор", "finance exchange — annual contract"),
    ("Бондаренко",        "finance exchange — Bondarenko reference"),
    ("Не слышал",         "finance exchange — Roman's follow-up question"),
    ("не согласован",     "finance exchange — approval status"),
    ("Русала",            "finance exchange — Rusal debt"),
]

# =========================================================================
# Data model
# =========================================================================

@dataclass
class Turn:
    start: float
    end: float
    speaker: str        # display name — real name for IT, id for ours
    text: str
    attribution_confidence: Optional[float] = None


# =========================================================================
# IT-team transcript parser
# =========================================================================

_TURN_RE = re.compile(
    r"^\s*\[(\d+):(\d+):(\d+\.\d+)\s*-\s*(\d+):(\d+):(\d+\.\d+)\]\s+([^:]+?):\s+(.*)$"
)


def parse_hms(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_it_team(path: Path) -> list[Turn]:
    """Parse DiarizerWhisper_130426.txt format.

    Each turn:
        [HH:MM:SS.mmm - HH:MM:SS.mmm] SPEAKER NAME: text...
    A leading speaker-header block is skipped by looking for the first
    line that matches the turn regex.
    """
    turns: list[Turn] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _TURN_RE.match(raw)
        if not m:
            continue
        h1, m1, s1, h2, m2, s2, spk, text = m.groups()
        turns.append(
            Turn(
                start=parse_hms(h1, m1, s1),
                end=parse_hms(h2, m2, s2),
                speaker=spk.strip(),
                text=text.strip(),
            )
        )
    return turns


# =========================================================================
# VERITAS aligned.json / aligned_v2*.json parser
# =========================================================================

def parse_veritas(path: Path) -> list[Turn]:
    """Parse either `aligned.json` or `aligned_v2_polished.json` format.
    Both have the same shape: {"segments": [...]} with fields
    start/end/text/speaker_id[/speaker_name][/attribution_confidence]."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    out: list[Turn] = []
    for s in segs:
        spk = s.get("speaker_name") or s.get("speaker_id") or s.get("speaker") or "SPEAKER_??"
        text = s.get("text", "") or ""
        if not text.strip():
            continue
        ac = s.get("attribution_confidence")
        out.append(
            Turn(
                start=float(s["start"]),
                end=float(s["end"]),
                speaker=str(spk),
                text=text.strip(),
                attribution_confidence=float(ac) if ac is not None else None,
            )
        )
    return out


def load_source(dir_or_file: Optional[Path], prefer: str = "original") -> tuple[list[Turn], Optional[str]]:
    """Resolve a source argument that can be either a JSON file or a
    directory containing aligned*.json.

    `prefer` chooses which file to load when multiple exist in a directory:
      * "original"  — aligned.json (the pipeline's output at time of archive
                      — crucially, pre-Phase-A+ for Apr-20 archives)
      * "repolished" — aligned_v2_polished.json (same ASR/diar bytes through
                       the current Phase-A+ postprocessor)
      * "raw"       — aligned_raw.json (pre-postprocessor)

    Returns (turns, loaded_filename). loaded_filename is useful for
    labelling the column in the HTML so the reader knows exactly what
    they're looking at.
    """
    if dir_or_file is None:
        return [], None
    p = dir_or_file
    if p.is_dir():
        cascades = {
            "original":   ["aligned.json", "aligned_v2_polished.json", "aligned_v2.json", "aligned_raw.json"],
            "repolished": ["aligned_v2_polished.json", "aligned_v2.json", "aligned.json", "aligned_raw.json"],
            "raw":        ["aligned_raw.json", "aligned.json", "aligned_v2.json", "aligned_v2_polished.json"],
        }
        for candidate in cascades.get(prefer, cascades["original"]):
            f = p / candidate
            if f.exists():
                print(f"[load] {p.name}/{candidate}  (prefer={prefer})", file=sys.stderr)
                return parse_veritas(f), candidate
        print(f"[warn] no aligned*.json found in {p}", file=sys.stderr)
        return [], None
    return parse_veritas(p), p.name


# =========================================================================
# Alignment: for each IT turn, find overlapping turns in other sources
# =========================================================================

def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def find_overlapping(anchor: Turn, others: list[Turn]) -> list[Turn]:
    """Return all `others` whose time range overlaps the anchor's."""
    return [o for o in others if overlaps(anchor.start, anchor.end, o.start, o.end)]


# =========================================================================
# Rendering
# =========================================================================

def fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def conf_badge(ac: Optional[float]) -> str:
    if ac is None:
        return ""
    if ac >= 0.8:
        cls = "ok"
    elif ac >= 0.6:
        cls = "warn"
    else:
        cls = "bad"
    return f'<span class="conf {cls}" title="attribution_confidence">{ac:.2f}</span>'


def render_cell(turns: list[Turn], flag_substrs: list[str]) -> str:
    if not turns:
        return '<td class="empty">—</td>'
    parts: list[str] = []
    for t in turns:
        text = html.escape(t.text)
        # Highlight target phrase substrings
        for sub, _label in flag_substrs:
            if sub.lower() in t.text.lower():
                # Case-insensitive replacement preserving original case
                pat = re.compile(re.escape(sub), re.IGNORECASE)
                text = pat.sub(lambda m: f'<mark>{html.escape(m.group(0))}</mark>', html.escape(t.text))
                break
        badge = conf_badge(t.attribution_confidence)
        parts.append(
            f'<div class="turn">'
            f'<div class="meta"><span class="spk">{html.escape(t.speaker)}</span>{badge}'
            f'<span class="ts">{fmt_ts(t.start)}–{fmt_ts(t.end)}</span></div>'
            f'<div class="text">{text}</div>'
            f'</div>'
        )
    return '<td>' + ''.join(parts) + '</td>'


def turn_has_any(t: Turn, substrs: list[tuple[str, str]]) -> bool:
    lower = t.text.lower()
    return any(sub.lower() in lower for sub, _ in substrs)


def render_target_callouts(
    it: list[Turn], old: list[Turn], fresh: list[Turn]
) -> str:
    """Render a pinned callout block for each target phrase showing how
    each source renders the surrounding moment."""
    sections: list[str] = []
    for sub, label in TARGET_PHRASES:
        anchors = [t for t in it if sub.lower() in t.text.lower()]
        if not anchors:
            sections.append(
                f'<div class="callout missing">'
                f'<div class="callout-title">Target: «{html.escape(sub)}» — <em>{html.escape(label)}</em></div>'
                f'<div class="callout-body">Not found in IT team transcript.</div>'
                f'</div>'
            )
            continue
        for anchor in anchors[:1]:  # only one instance per phrase to keep the header compact
            old_match = find_overlapping(anchor, old)
            fresh_match = find_overlapping(anchor, fresh)
            sections.append(
                f'<div class="callout">'
                f'<div class="callout-title">Target: «{html.escape(sub)}» — <em>{html.escape(label)}</em> '
                f'(@{fmt_ts(anchor.start)})</div>'
                f'<table class="callout-table"><thead>'
                f'<tr><th>IT team</th><th>Ours — old pipeline (archive)</th><th>Ours — current pipeline (fresh E2E)</th></tr></thead>'
                f'<tbody><tr>'
                f'{render_cell([anchor], TARGET_PHRASES)}'
                f'{render_cell(old_match, TARGET_PHRASES)}'
                f'{render_cell(fresh_match, TARGET_PHRASES)}'
                f'</tr></tbody></table>'
                f'</div>'
            )
    return '\n'.join(sections)


def count_unique_speakers(turns: list[Turn]) -> int:
    return len({t.speaker for t in turns})


def header_stats(
    it: list[Turn], old: list[Turn], fresh: list[Turn]
) -> str:
    rows = [
        ("Turns",     len(it), len(old), len(fresh)),
        ("Unique speakers", count_unique_speakers(it), count_unique_speakers(old), count_unique_speakers(fresh)),
        ("Total duration (s)",
            round(max((t.end for t in it), default=0), 1),
            round(max((t.end for t in old), default=0), 1),
            round(max((t.end for t in fresh), default=0), 1)),
    ]
    tr = "".join(
        f'<tr><th>{k}</th><td>{a}</td><td>{b}</td><td>{c}</td></tr>'
        for k, a, b, c in rows
    )
    return (
        f'<table class="stats"><thead>'
        f'<tr><th></th><th>IT team</th><th>Ours — old pipeline (archive)</th><th>Ours — current pipeline (fresh E2E)</th></tr>'
        f'</thead><tbody>{tr}</tbody></table>'
    )


CSS = """
body { font: 13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin: 0; padding: 20px 32px; color: #222; background:#fafafa; }
h1 { font: 700 22px Georgia,serif; color:#B2001F; margin: 0 0 4px; }
.subtitle { color:#666; margin-bottom: 16px; }
.stats { border-collapse:collapse; margin: 12px 0 20px; background:white; }
.stats th, .stats td { border:1px solid #ddd; padding:6px 12px; text-align:left; }
.stats thead { background:#f0f0f0; }
.callout { background:white; border-left: 4px solid #B2001F; padding: 10px 16px; margin: 10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.callout.missing { border-left-color: #999; }
.callout-title { font-weight:600; margin-bottom:8px; }
.callout-body { color:#888; font-style:italic; }
.callout-table { width:100%; border-collapse:collapse; table-layout:fixed; }
.callout-table th, .callout-table td { border:1px solid #e0e0e0; padding:8px; vertical-align:top; width:33.33%; }
.callout-table thead th { background:#f7f3f4; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:#B2001F; }
.main { border-collapse:collapse; width:100%; table-layout:fixed; background:white; margin-top:20px; }
.main th, .main td { border:1px solid #e0e0e0; padding:8px; vertical-align:top; width:33.33%; }
.main thead th { position:sticky; top:0; background:#f0f0f0; z-index:1; padding:10px; font-weight:600; }
.turn { border-bottom: 1px dashed #eee; padding: 4px 0; }
.turn:last-child { border-bottom:none; }
.meta { display:flex; gap:8px; font-size:10.5px; color:#888; align-items:center; margin-bottom:3px; }
.spk { font-weight:600; color:#B2001F; }
.ts { color:#999; margin-left:auto; font-variant-numeric: tabular-nums; }
.conf { font-size:10px; padding:1px 5px; border-radius:3px; font-variant-numeric:tabular-nums; }
.conf.ok { background:#d8f0d8; color:#265d26; }
.conf.warn { background:#fbe6c5; color:#7a4b07; }
.conf.bad { background:#f8c9c9; color:#7a0a0a; }
.empty { color:#bbb; text-align:center; }
mark { background:#fff3a8; padding:0 2px; border-radius:2px; }
"""


def render_main_table(
    it: list[Turn],
    old: list[Turn],
    fresh: list[Turn],
    it_label: str = "IT team (Whisper + pyannote/?)",
    old_label: str = "Ours — old pipeline (archive)",
    fresh_label: str = "Ours — current pipeline (fresh E2E)",
) -> str:
    rows: list[str] = []
    for anchor in it:
        old_match = find_overlapping(anchor, old)
        fresh_match = find_overlapping(anchor, fresh)
        any_flag = (
            turn_has_any(anchor, TARGET_PHRASES)
            or any(turn_has_any(t, TARGET_PHRASES) for t in old_match + fresh_match)
        )
        row_cls = ' class="flagged"' if any_flag else ""
        rows.append(
            f'<tr{row_cls}>'
            f'{render_cell([anchor], TARGET_PHRASES)}'
            f'{render_cell(old_match, TARGET_PHRASES)}'
            f'{render_cell(fresh_match, TARGET_PHRASES)}'
            f'</tr>'
        )
    return (
        f'<table class="main"><thead>'
        f'<tr><th>{html.escape(it_label)}</th><th>{html.escape(old_label)}</th><th>{html.escape(fresh_label)}</th></tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table>'
    )


def build_html(
    it: list[Turn],
    old: list[Turn],
    fresh: list[Turn],
    meta: dict,
    title: str = "Админ 13-04-2026 — three-column comparison",
    subtitle: str = "IT team's Whisper+Gemma stack · our old pre-Phase-A+ output · our fresh current-pipeline run",
    it_label: str = "IT team (Whisper + pyannote/?)",
    old_label: str = "Ours — old pipeline (archive)",
    fresh_label: str = "Ours — current pipeline (fresh E2E)",
) -> str:
    meta_lines = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(str(v))}</li>" for k, v in meta.items())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="subtitle">{html.escape(subtitle)}</div>
<ul class="meta">{meta_lines}</ul>
{header_stats(it, old, fresh)}
<h2>Target-passage callouts</h2>
{render_target_callouts(it, old, fresh)}
<h2>Full timeline (anchor: IT team turns)</h2>
{render_main_table(it, old, fresh, it_label, old_label, fresh_label)}
</body>
</html>
"""


# =========================================================================
# CLI
# =========================================================================

def resolve_latest_fresh() -> Optional[Path]:
    """Read benchmarks/data/admin_13_04_fresh_LATEST.txt written by the
    E2E runner. Returns the fresh directory if the marker exists."""
    marker = REPO_ROOT / "benchmarks" / "data" / "admin_13_04_fresh_LATEST.txt"
    if not marker.exists():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    if not name:
        return None
    target = REPO_ROOT / "benchmarks" / "data" / name
    return target if target.exists() else None


def default_old_archive() -> Optional[Path]:
    """The user's reviewed old archive — pre-Phase-A+."""
    p = REPO_ROOT / "data" / "archive" / "2026-04-20_Админ_13-04-2026_без_мусора_в_начале"
    return p if p.exists() else None


def default_it_transcript() -> Optional[Path]:
    """Look for the IT team's transcript in a few plausible places."""
    candidates = [
        REPO_ROOT / "data" / "reference" / "DiarizerWhisper_130426_with_speakers.txt",
        REPO_ROOT / "benchmarks" / "data" / "DiarizerWhisper_130426 with speakers.txt",
        REPO_ROOT / "benchmarks" / "data" / "it_team" / "DiarizerWhisper_130426 with speakers.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def estimate_it_offset(it: list[Turn], ours: list[Turn]) -> float:
    """Heuristically estimate how much earlier our trimmed audio starts
    vs the IT team's untrimmed audio. We assume both recordings cover
    roughly the same meeting content and that the first minute of real
    discussion appears in both. Offset = first_IT_turn.start - first_our_turn.start.

    This is a ROUGH first-pass estimate. If the user reports the offset
    looks wrong, they can override with --it-offset SECONDS."""
    if not it or not ours:
        return 0.0
    it_first = it[0].start
    our_first = ours[0].start
    delta = it_first - our_first
    # Sanity floor/ceiling: we only care when IT is clearly later (>30s)
    # and the offset is within a reasonable meeting-length range.
    if delta < 30.0 or delta > 3600.0:
        return 0.0
    return delta


def apply_offset(turns: list[Turn], offset: float) -> list[Turn]:
    """Return a new list with `offset` subtracted from every start/end.
    IT team timestamps include pre-meeting chatter that was trimmed from
    our audio, so we shift them earlier to line up the same moments."""
    if not offset:
        return turns
    return [
        Turn(
            start=max(0.0, t.start - offset),
            end=max(0.0, t.end - offset),
            speaker=t.speaker,
            text=t.text,
            attribution_confidence=t.attribution_confidence,
        )
        for t in turns
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--it-team", type=Path, default=None,
                    help="Path to IT team's DiarizerWhisper transcript (.txt)")
    ap.add_argument("--old", type=Path, default=None,
                    help="Path to old archive dir or aligned*.json file")
    ap.add_argument("--old-variant", choices=["original", "repolished", "raw"],
                    default="original",
                    help="Which archive file to prefer. 'original' (default) loads "
                         "aligned.json (shows pre-Phase-A+ pipeline output for Apr-20 archives). "
                         "'repolished' loads aligned_v2_polished.json (same bytes through new "
                         "postprocessor). 'raw' loads aligned_raw.json (pre-postprocessor).")
    ap.add_argument("--fresh", type=Path, default=None,
                    help="Path to fresh E2E dir. Defaults to admin_13_04_fresh_LATEST marker.")
    ap.add_argument("--it-offset", type=float, default=None,
                    help="Seconds to subtract from IT team timestamps. If omitted, "
                         "auto-estimated from first-turn delta. Use 0 to disable.")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "benchmarks" / "reports" / "admin_13_04_compare.html",
                    help="Output HTML path")
    ap.add_argument("--title", type=str, default=None,
                    help="Page title (and <h1>). Default describes three-column pipeline comparison.")
    ap.add_argument("--subtitle", type=str, default=None,
                    help="Subtitle beneath the page title.")
    ap.add_argument("--it-label", type=str, default=None,
                    help="Column header for the IT-team transcript. "
                         "Default: 'IT team (Whisper + pyannote/?)'.")
    ap.add_argument("--old-label", type=str, default=None,
                    help="Column header for the old/archive column. "
                         "Default: 'Ours — old pipeline (archive)'. Use this when "
                         "comparing pipeline variants (e.g. 'Ours — GigaAM + pyannote 4.0').")
    ap.add_argument("--fresh-label", type=str, default=None,
                    help="Column header for the fresh E2E column. Default: "
                         "'Ours — current pipeline (fresh E2E)'. Use this when "
                         "comparing pipelines (e.g. 'Ours — GigaAM + pyannote 4.0 + T-Pro 2.0').")
    args = ap.parse_args()

    it_path = args.it_team or default_it_transcript()
    if it_path is None or not it_path.exists():
        print("ERROR: IT-team transcript not found. Use --it-team to specify.", file=sys.stderr)
        print("  Checked: data/reference/, benchmarks/data/, benchmarks/data/it_team/", file=sys.stderr)
        return 2

    old_path = args.old or default_old_archive()
    fresh_path = args.fresh or resolve_latest_fresh()

    print(f"IT team:  {it_path}", file=sys.stderr)
    print(f"Old:      {old_path}", file=sys.stderr)
    print(f"Fresh:    {fresh_path}", file=sys.stderr)

    it_raw = parse_it_team(it_path)
    # For the archive column, load the variant the user asked for.
    # For the fresh column, always load aligned.json (the live orchestrator
    # output) — fresh dirs never have aligned_v2* files.
    old_turns, old_filename = load_source(old_path, prefer=args.old_variant)
    fresh_turns, fresh_filename = load_source(fresh_path, prefer="original")

    # Estimate offset using fresh turns as our reference (or archive if fresh missing)
    reference = fresh_turns or old_turns
    if args.it_offset is None:
        offset = estimate_it_offset(it_raw, reference)
        auto_flag = "auto-estimated"
    else:
        offset = args.it_offset
        auto_flag = "user-specified"


    print(f"[offset] IT->ours: {offset:.1f}s ({auto_flag})", file=sys.stderr)
    if it_raw and reference:
        print(f"         first IT turn: {fmt_ts(it_raw[0].start)}  "
              f"first ours turn: {fmt_ts(reference[0].start)}", file=sys.stderr)

    it = apply_offset(it_raw, offset)

    print(f"[load] IT:     {len(it)} turns (offset {offset:.1f}s applied)", file=sys.stderr)
    print(f"[load] Old:    {len(old_turns)} turns  [{old_filename}]", file=sys.stderr)
    print(f"[load] Fresh:  {len(fresh_turns)} turns  [{fresh_filename}]", file=sys.stderr)

    # Human-readable variant description
    variant_desc = {
        "original":   "aligned.json - pipeline output as it was when archived "
                      "(pre-Phase-A+ for Apr-20 archives)",
        "repolished": "aligned_v2_polished.json - same ASR/diar bytes re-run "
                      "through the current Phase-A+ postprocessor",
        "raw":        "aligned_raw.json - pre-postprocessor output",
    }

    meta = {
        "IT-team transcript": it_path.name,
        "IT->ours offset applied": f"-{offset:.1f}s ({auto_flag})",
        "Archive file loaded": f"{old_filename or '(missing)'} ({variant_desc.get(args.old_variant, '')})",
        "Fresh file loaded": fresh_filename or "(missing - run run_admin_e2e.py first)",
    }

    # Build human-readable title + subtitle, override-able via CLI
    default_title = "Admin 13-04-2026 - three-column comparison"
    default_subtitle = (
        "IT team's Whisper+Gemma stack | our old pre-Phase-A+ output | "
        "our fresh current-pipeline run"
    )
    title = args.title if args.title else default_title
    subtitle = args.subtitle if args.subtitle else default_subtitle

    it_label = args.it_label or "IT team (Whisper + pyannote/?)"
    old_label = args.old_label or "Ours - old pipeline (archive)"
    fresh_label = args.fresh_label or "Ours - current pipeline (fresh E2E)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_html(
            it,
            old_turns,
            fresh_turns,
            meta,
            title=title,
            subtitle=subtitle,
            it_label=it_label,
            old_label=old_label,
            fresh_label=fresh_label,
        ),
        encoding="utf-8",
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
