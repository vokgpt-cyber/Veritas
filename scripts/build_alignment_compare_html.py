#!/usr/bin/env python3
"""Build a side-by-side HTML view of old vs new aligner output.

Renders up to three parallel columns:
  - LEFT: baseline aligned_raw.json (or aligned.json) produced by the
    pre-fix aligner (max-single-overlap rule, no sentence splitting).
  - MIDDLE: aligned_v2.json produced by the new aligner (sum-aggregate rule,
    sentence-boundary pre-splitting, attribution_confidence populated).
  - RIGHT (optional): aligned_v2_polished.json -- what the user actually sees,
    after postprocess_transcript merges same-speaker turns and applies the
    gap-aware seam-polish rules (punctuation + capitalization at seams).

Each segment is rendered as a card: [MM:SS] SPEAKER -- text. Speakers get
stable colors across all columns so you can eyeball which chunks got
re-attributed. The new + polished columns render a confidence badge and
flag contested (<0.6) segments in yellow.

A TOP banner pins the target phrase, extracted from each column with
surrounding context, so you can see the mega-block next to the tight
isolation the new aligner produced, and what the reader-facing polished
output looks like.

Usage:
    python scripts/build_alignment_compare_html.py \
        --archive "data/archive/2026-04-20_..." \
        --out benchmarks/reports/alignment_compare.html

If aligned_v2_polished.json exists in the archive, it is auto-loaded and
a third column is rendered. Pass --no-polished to suppress.

No model deps; pure JSON + HTML.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

EPAM_RED = "#B2001F"

# Palette for up to 20 speakers; stable across all columns by speaker_id.
SPEAKER_COLORS = [
    "#B2001F", "#2E5EAA", "#3F7D3F", "#A05300", "#6B3FA0",
    "#0E7C86", "#B84A82", "#3A3A3A", "#7A5E00", "#046C45",
    "#8F3A84", "#1E6091", "#8B3A3A", "#4E6E58", "#6A4E0F",
    "#5A3F8F", "#2B5F75", "#6E1E3A", "#2D5F3D", "#514A00",
]


def speaker_color(spk: str) -> str:
    if not spk:
        return "#555"
    idx = 0
    for ch in spk:
        if ch.isdigit():
            idx = idx * 10 + int(ch)
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


def fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    mm = int(t // 60)
    ss = int(t % 60)
    return f"{mm}:{ss:02d}"


def load_segments(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    return segs


def extract_target_window(segments: list[dict], phrase_markers: list[str],
                          before: int = 2, after: int = 3) -> list[dict]:
    hit_indices = []
    for i, s in enumerate(segments):
        text = s.get("text", "")
        for m in phrase_markers:
            if m in text:
                hit_indices.append(i)
                break
    if not hit_indices:
        return []
    keep = set()
    for i in hit_indices:
        for j in range(max(0, i - before), min(len(segments), i + after + 1)):
            keep.add(j)
    return [segments[i] for i in sorted(keep)]


def render_segment_card(seg: dict, is_target: bool = False,
                        show_confidence: bool = False) -> str:
    spk = seg.get("speaker_id") or seg.get("speaker") or "?"
    text = seg.get("text", "")
    start = seg.get("start", 0.0)
    end = seg.get("end", 0.0)
    dur = end - start
    conf = seg.get("attribution_confidence")
    color = speaker_color(spk)

    badges = []
    if show_confidence and conf is not None:
        if conf >= 0.8:
            badge_bg = "#E8F5E9"; badge_fg = "#1B5E20"
        elif conf >= 0.6:
            badge_bg = "#FFF8E1"; badge_fg = "#795548"
        else:
            badge_bg = "#FFEBEE"; badge_fg = "#B71C1C"
        badges.append(
            f'<span class="badge" style="background:{badge_bg};color:{badge_fg};">'
            f'conf {conf:.2f}</span>'
        )
    badges.append(f'<span class="badge dur">{dur:.1f}s</span>')

    classes = ["seg"]
    if is_target:
        classes.append("target")
    if show_confidence and conf is not None and conf < 0.6:
        classes.append("contested")

    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="seghead">'
        f'<span class="ts">[{fmt_time(start)}-{fmt_time(end)}]</span> '
        f'<span class="spk" style="color:{color};">{escape(spk)}</span>'
        f' {"".join(badges)}'
        f'</div>'
        f'<div class="segtext">{escape(text)}</div>'
        f'</div>'
    )


def mark_target(segments: list[dict], phrase_markers: list[str]) -> list[bool]:
    out = []
    for s in segments:
        t = s.get("text", "")
        out.append(any(m in t for m in phrase_markers))
    return out


def compute_stats(segments: list[dict]) -> dict:
    total_dur = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
    confs = [s.get("attribution_confidence") for s in segments
             if s.get("attribution_confidence") is not None]
    spk_dur: dict[str, float] = {}
    for s in segments:
        spk = s.get("speaker_id") or s.get("speaker") or "?"
        spk_dur[spk] = spk_dur.get(spk, 0.0) + (s.get("end", 0) - s.get("start", 0))

    return {
        "n_segments": len(segments),
        "n_speakers": len(spk_dur),
        "total_duration": total_dur,
        "avg_segment_duration": total_dur / len(segments) if segments else 0.0,
        "longest_segment": max(
            ((s.get("end", 0) - s.get("start", 0)) for s in segments),
            default=0.0,
        ),
        "n_conf": len(confs),
        "avg_conf": sum(confs) / len(confs) if confs else None,
        "contested": sum(1 for c in confs if c < 0.6),
        "spk_share": {k: (100 * v / total_dur) if total_dur else 0.0
                      for k, v in spk_dur.items()},
    }


def render_stats_panel(old: dict, new: dict, polished: dict | None = None) -> str:
    def fmt_conf(s: dict) -> str:
        if s.get("avg_conf") is None:
            return "--"
        return f"{s['avg_conf']:.2f} avg ({s['contested']}/{s['n_conf']} &lt; 0.6)"

    cols = [("Old aligner", old), ("New aligner", new)]
    if polished is not None:
        cols.append(("+ Polished", polished))

    rows = []
    rows.append(("Segments", [str(s["n_segments"]) for _, s in cols]))
    rows.append(("Speakers detected", [str(s["n_speakers"]) for _, s in cols]))
    rows.append(("Avg segment duration",
                 [f"{s['avg_segment_duration']:.1f}s" for _, s in cols]))
    rows.append(("Longest segment",
                 [f"{s['longest_segment']:.1f}s" for _, s in cols]))
    conf_cells = []
    for name, s in cols:
        conf_cells.append("--" if name == "Old aligner" else fmt_conf(s))
    rows.append(("Attribution confidence", conf_cells))

    thead_cells = "".join(f"<th>{escape(name)}</th>" for name, _ in cols)
    tbody = "\n".join(
        f'<tr><th>{escape(k)}</th>'
        + "".join(f'<td>{v}</td>' for v in vals)
        + '</tr>'
        for k, vals in rows
    )
    return (
        '<div class="stats">'
        '<h2>Summary</h2>'
        '<table><thead>'
        f'<tr><th></th>{thead_cells}</tr>'
        '</thead><tbody>'
        f'{tbody}'
        '</tbody></table>'
        '</div>'
    )


def render_speaker_share(old: dict, new: dict, polished: dict | None = None) -> str:
    keys = set(old["spk_share"]) | set(new["spk_share"])
    if polished is not None:
        keys |= set(polished["spk_share"])
    all_spks = sorted(keys)
    rows = []
    for spk in all_spks:
        o = old["spk_share"].get(spk, 0.0)
        n = new["spk_share"].get(spk, 0.0)
        delta = n - o
        color = speaker_color(spk)
        d_color = "#2E7D32" if abs(delta) < 0.5 else ("#B71C1C" if abs(delta) > 2.0 else "#795548")
        extra = ""
        if polished is not None:
            p = polished["spk_share"].get(spk, 0.0)
            extra = f'<td class="num">{p:.1f}%</td>'
        rows.append(
            f'<tr>'
            f'<td><span class="spkdot" style="background:{color};"></span>'
            f'{escape(spk)}</td>'
            f'<td class="num">{o:.1f}%</td>'
            f'<td class="num">{n:.1f}%</td>'
            f'{extra}'
            f'<td class="num" style="color:{d_color};">{delta:+.1f}%</td>'
            f'</tr>'
        )
    polished_header = '<th>Polished</th>' if polished is not None else ''
    return (
        '<div class="share">'
        '<h2>Speaker share (% of aligned time)</h2>'
        '<table><thead>'
        f'<tr><th>Speaker</th><th>Old</th><th>New</th>{polished_header}<th>D (new - old)</th></tr>'
        '</thead><tbody>'
        + "\n".join(rows) +
        '</tbody></table>'
        '</div>'
    )


def render_columns(old_segs: list[dict], new_segs: list[dict],
                   phrase_markers: list[str],
                   polished_segs: list[dict] | None = None) -> str:
    old_target = mark_target(old_segs, phrase_markers)
    new_target = mark_target(new_segs, phrase_markers)

    left = "\n".join(
        render_segment_card(s, is_target=old_target[i], show_confidence=False)
        for i, s in enumerate(old_segs)
    )
    middle = "\n".join(
        render_segment_card(s, is_target=new_target[i], show_confidence=True)
        for i, s in enumerate(new_segs)
    )

    polished_html = ""
    if polished_segs is not None:
        p_target = mark_target(polished_segs, phrase_markers)
        polished_html_body = "\n".join(
            render_segment_card(s, is_target=p_target[i], show_confidence=True)
            for i, s in enumerate(polished_segs)
        )
        polished_html = (
            f'<div class="col"><h2>+ Polished ({len(polished_segs)} turns)</h2>'
            '<div class="coldesc">Same-speaker merge | gap-aware seam polish | reader-facing</div>'
            f'{polished_html_body}</div>'
        )

    cols_class = "cols cols-3" if polished_segs is not None else "cols"
    return (
        f'<div class="{cols_class}">'
        f'<div class="col"><h2>Old aligner ({len(old_segs)} segs)</h2>'
        '<div class="coldesc">Max single-segment overlap | no sentence splitting</div>'
        f'{left}</div>'
        f'<div class="col"><h2>New aligner ({len(new_segs)} segs)</h2>'
        '<div class="coldesc">Sum-aggregate overlap | sentence-boundary splitting | attribution confidence</div>'
        f'{middle}</div>'
        f'{polished_html}'
        '</div>'
    )


def render_target_callout(old_segs: list[dict], new_segs: list[dict],
                          phrase_markers: list[str],
                          polished_segs: list[dict] | None = None) -> str:
    old_window = extract_target_window(old_segs, phrase_markers, before=1, after=2)
    new_window = extract_target_window(new_segs, phrase_markers, before=2, after=3)
    polished_window = (
        extract_target_window(polished_segs, phrase_markers, before=1, after=2)
        if polished_segs is not None else []
    )
    if not old_window and not new_window and not polished_window:
        return ""
    old_target = mark_target(old_window, phrase_markers)
    new_target = mark_target(new_window, phrase_markers)
    old_html = "\n".join(
        render_segment_card(s, is_target=old_target[i], show_confidence=False)
        for i, s in enumerate(old_window)
    )
    new_html = "\n".join(
        render_segment_card(s, is_target=new_target[i], show_confidence=True)
        for i, s in enumerate(new_window)
    )

    polished_section = ""
    if polished_segs is not None:
        p_target = mark_target(polished_window, phrase_markers)
        polished_html_body = "\n".join(
            render_segment_card(s, is_target=p_target[i], show_confidence=True)
            for i, s in enumerate(polished_window)
        )
        polished_section = (
            f'<div class="col"><h3>+ Polished ({len(polished_window)} turns in window)</h3>'
            f'{polished_html_body}</div>'
        )

    cols_class = "cols cols-3" if polished_segs is not None else "cols"
    return (
        '<div class="callout">'
        '<h2>Target case: <span class="phrase">Sasha, ne govori phrase</span></h2>'
        '<p class="co-desc">The interjection was buried in a ~20s mega-block in the old aligner. '
        'The new aligner isolates it to its own ~1.7s segment. The polished column shows what '
        'the reader sees after same-speaker merge: the isolation survives because it is '
        'bookended by a different-speaker turn. Attribution is now a diarization/voiceprint '
        'problem, not an alignment problem. See the confidence badge.</p>'
        f'<div class="{cols_class}">'
        f'<div class="col"><h3>Old ({len(old_window)} segs in window)</h3>{old_html}</div>'
        f'<div class="col"><h3>New ({len(new_window)} segs in window)</h3>{new_html}</div>'
        f'{polished_section}'
        '</div>'
        '</div>'
    )


CSS = """
* { box-sizing: border-box; }
body {
  font-family: 'Arial Narrow', 'Arial', sans-serif;
  margin: 0; padding: 0; background: #F5F4F0; color: #1A1A1A;
}
header {
  background: %(red)s; color: #fff; padding: 20px 32px;
  display: flex; align-items: baseline; gap: 16px;
  border-bottom: 4px solid #1A1A1A;
}
header h1 {
  font-family: Georgia, serif; font-weight: normal; font-size: 26px; margin: 0;
  letter-spacing: 0.5px;
}
header .sub { opacity: 0.85; font-size: 15px; }
main { padding: 24px 32px 64px; max-width: 2100px; margin: 0 auto; }
h2 {
  font-family: Georgia, serif; font-weight: normal; font-size: 20px;
  margin: 20px 0 8px; color: %(red)s; letter-spacing: 0.3px;
}
h3 {
  font-family: Georgia, serif; font-weight: normal; font-size: 17px;
  margin: 12px 0 6px; color: #333;
}
.stats, .share, .callout {
  background: #fff; border: 1px solid #E0DDD5;
  padding: 16px 20px; margin-bottom: 20px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.callout .phrase {
  color: %(red)s; font-weight: bold; font-family: Georgia, serif;
}
.co-desc { color: #555; font-size: 13px; line-height: 1.45; margin: 4px 0 14px; }
table { border-collapse: collapse; width: 100%%; font-size: 13px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #EEE; }
th { background: #FAFAF7; color: #666; font-weight: normal; font-size: 12px; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.spkdot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%%;
  margin-right: 8px; vertical-align: middle;
}
.cols {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
.cols.cols-3 {
  grid-template-columns: 1fr 1fr 1fr;
}
.col { background: #fff; border: 1px solid #E0DDD5; padding: 14px 16px;
       overflow-y: auto; max-height: 85vh; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.col h2 { margin-top: 0; }
.coldesc { color: #777; font-size: 12px; margin-bottom: 10px; line-height: 1.4; }
.seg {
  border-left: 3px solid #DDD; padding: 6px 10px; margin: 6px 0;
  background: #FBFBF9; border-radius: 0 2px 2px 0;
}
.seg.target { background: #FFF4E1; border-left-color: %(red)s; }
.seg.contested { background: #FFEBEE; border-left-color: #B71C1C; }
.seghead {
  font-size: 11px; color: #777; margin-bottom: 4px; display: flex;
  align-items: center; gap: 6px; flex-wrap: wrap;
}
.ts { font-variant-numeric: tabular-nums; color: #999; font-family: monospace; }
.spk { font-weight: bold; font-size: 12px; }
.badge {
  display: inline-block; padding: 1px 6px; font-size: 10px;
  border-radius: 8px; background: #EEE; color: #444; font-family: monospace;
}
.badge.dur { background: transparent; color: #BBB; padding-right: 0; }
.segtext { font-size: 14px; line-height: 1.5; color: #222; }
footer {
  text-align: center; color: #999; font-size: 11px; padding: 20px;
  border-top: 1px solid #E0DDD5; margin-top: 40px;
}
""" % {"red": EPAM_RED}


def build_html(archive: Path, title: str, phrase_markers: list[str],
               include_polished: bool = True) -> str:
    old_path = archive / "aligned_raw.json"
    if not old_path.exists():
        old_path = archive / "aligned.json"
    new_path = archive / "aligned_v2.json"
    polished_path = archive / "aligned_v2_polished.json"

    old_segs = load_segments(old_path)
    new_segs = load_segments(new_path)
    polished_segs = None
    if include_polished and polished_path.exists():
        polished_segs = load_segments(polished_path)

    old_stats = compute_stats(old_segs)
    new_stats = compute_stats(new_segs)
    polished_stats = compute_stats(polished_segs) if polished_segs is not None else None

    sub = "Old (max-overlap, no split) vs New (sum-aggregate, sentence-split)"
    if polished_segs is not None:
        sub += " vs Polished (same-speaker merge, seam polish)"

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>VERITAS | Aligner A/B</h1>
  <span class="sub">{escape(sub)}</span>
  <span class="sub" style="margin-left:auto;">{escape(archive.name)}</span>
</header>
<main>
  {render_stats_panel(old_stats, new_stats, polished_stats)}
  {render_speaker_share(old_stats, new_stats, polished_stats)}
  {render_target_callout(old_segs, new_segs, phrase_markers, polished_segs)}
  <h2>Full side-by-side</h2>
  {render_columns(old_segs, new_segs, phrase_markers, polished_segs)}
</main>
<footer>EPAM VERITAS -- {escape(archive.name)} -- generated 2026-04-21</footer>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", type=str,
                   default="VERITAS aligner A/B")
    p.add_argument("--phrase", action="append", default=None,
                   help="Target phrase marker (repeat).")
    p.add_argument("--no-polished", dest="polished", action="store_false",
                   help="Suppress the polished column even if aligned_v2_polished.json exists.")
    p.set_defaults(polished=True)
    args = p.parse_args()

    markers = args.phrase or [
        "\u0421\u0430\u0448, \u043d\u0435 \u0433\u043e\u0432\u043e\u0440\u0438",
        "\u043d\u0435 \u0433\u043e\u0432\u043e\u0440\u0438, \u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430",
    ]

    html = build_html(args.archive, args.title, markers,
                      include_polished=args.polished)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
