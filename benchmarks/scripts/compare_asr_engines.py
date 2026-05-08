"""Side-by-side comparison of two ASR engines against a gold reference.

Runs the full VERITAS pipeline twice on the same audio — once with
each engine — then scores both against the gold transcript and
generates a 3-column HTML report:

    +-----------+----------+------------+
    |   Gold    | Engine A | Engine B   |
    +-----------+----------+------------+
    | turn 1    | seg X    | seg Y      |
    | turn 2    | seg X+1  | seg Y+1    |
    | ...                              |
    +-----------+----------+------------+

Plus a header panel with WER/CER for each engine and aggregate stats
(speakers detected, segment counts, runtime).

The pipeline runs are full E2E — same orchestrator, same diarization,
same post-processing. Only the ASR engine differs. This isolates ASR
quality from everything else.

Default config: gold reference is court_hearing_129 (281 turns), audio
is `benchmarks/audio_cache/Court hearing 129_01_16k_mono.wav`. Override
via CLI flags. Summarization is disabled for the comparison runs to
save 5-10 min per side; the protocol output is irrelevant for ASR
quality scoring.

Usage:
    python benchmarks/scripts/compare_asr_engines.py
    python benchmarks/scripts/compare_asr_engines.py --engines gigaam whisperx
    python benchmarks/scripts/compare_asr_engines.py --skip-runs    # use existing outputs

Outputs:
    benchmarks/reports/asr_compare_<stamp>.html — open this in a browser
    benchmarks/reports/asr_compare_<stamp>.json — raw stats for CI
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# noqa: E402
from benchmarks.scripts.score_wer import score as score_wer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("asr_compare")

DEFAULT_AUDIO = REPO_ROOT / "benchmarks" / "audio_cache" / "Court hearing 129_01_16k_mono.wav"
DEFAULT_GOLD = REPO_ROOT / "benchmarks" / "gold" / "court_hearing_129.jsonl"


# =====================================================================
# Loaders
# =====================================================================

def load_gold_jsonl(path: Path) -> list[dict]:
    """Read the gold JSONL into a list of {turn, speaker, text} dicts."""
    if not path.exists():
        logger.error("Gold reference not found: %s", path)
        sys.exit(1)
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping bad gold line: %s (%s)", line[:80], exc)
            continue
        out.append(entry)
    logger.info("Gold loaded: %d turns from %s", len(out), path.name)
    return out


def load_aligned_json(path: Path) -> list[dict]:
    """Load aligned.json from a benchmarks/data/ run folder.

    Schema: list of {start, end, text, speaker_id, speaker_name,
    confidence, attribution_confidence}.
    """
    if not path.exists():
        logger.error("aligned.json not found: %s", path)
        sys.exit(2)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        raw = raw["segments"]
    if not isinstance(raw, list):
        logger.error("Unexpected aligned.json shape: %r", type(raw))
        sys.exit(3)
    return raw


def latest_run_dir(prefix: str) -> Optional[Path]:
    """Return the latest benchmarks/data/<prefix>_<stamp>/ folder."""
    base = REPO_ROOT / "benchmarks" / "data"
    candidates = sorted(
        base.glob(f"{prefix}_*"),
        key=lambda p: p.name,
        reverse=True,
    )
    for c in candidates:
        if c.is_dir() and (c / "aligned.json").exists():
            return c
    return None


# =====================================================================
# Pipeline runner
# =====================================================================

def run_pipeline(
    audio: Path,
    engine: str,
    archive_prefix: str,
) -> Path:
    """Invoke run_admin_e2e.py with --engine <X> --archive-prefix <Y>.

    Returns the resulting archive folder path.
    """
    cmd = [
        sys.executable,
        str(HERE / "run_admin_e2e.py"),
        "--audio", str(audio),
        "--meeting-type", "generic",  # ASR comparison; no protocol prompt
        "--engine", engine,
        "--archive-prefix", archive_prefix,
        "--skip-summarization",  # ASR quality only — skip 5-10 min LLM stage
        "--skip-ollama-check",
    ]
    logger.info("Running pipeline (engine=%s): %s", engine, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        logger.error("Pipeline failed (engine=%s, exit code=%d)", engine, proc.returncode)
        sys.exit(proc.returncode)

    folder = latest_run_dir(archive_prefix)
    if folder is None:
        logger.error("No run folder found for prefix %s", archive_prefix)
        sys.exit(4)
    logger.info("Pipeline run (engine=%s) -> %s", engine, folder.name)
    return folder


# =====================================================================
# Scoring + concat helpers
# =====================================================================

def turns_to_text(items: list[dict], text_key: str = "text") -> str:
    """Concatenate per-turn text fields with single spaces between."""
    parts = [str(i.get(text_key, "")).strip() for i in items]
    return " ".join(p for p in parts if p)


def score_engine(gold_text: str, engine_text: str) -> dict:
    """Compute WER/CER + edit metrics for one engine vs gold."""
    return score_wer(gold_text, engine_text, skip_cer=False)


# =====================================================================
# HTML report
# =====================================================================

CSS = """
:root {
  --epam-red: #B2001F;
  --gray-900: #111;
  --gray-700: #444;
  --gray-500: #777;
  --gray-300: #ccc;
  --gray-100: #f3f3f3;
  --green: #2e7d32;
  --amber: #b07000;
  --red: #b00020;
}
body {
  font-family: 'Arial Narrow', Arial, sans-serif;
  font-size: 13px;
  color: var(--gray-900);
  margin: 0;
  padding: 24px;
  background: #fafafa;
}
h1 {
  font-family: Georgia, serif;
  font-size: 24px;
  color: var(--epam-red);
  margin: 0 0 8px;
}
.subtitle { color: var(--gray-500); margin-bottom: 24px; }

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.stat-card {
  background: white;
  border: 1px solid var(--gray-300);
  border-radius: 8px;
  padding: 16px;
}
.stat-card h3 {
  font-family: Georgia, serif;
  margin: 0 0 12px;
  font-size: 16px;
  color: var(--epam-red);
}
.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px dashed var(--gray-300);
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--gray-700); }
.stat-value { font-weight: bold; color: var(--gray-900); font-variant-numeric: tabular-nums; }
.wer-good { color: var(--green); }
.wer-amber { color: var(--amber); }
.wer-bad { color: var(--red); }

.transcript-table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  border: 1px solid var(--gray-300);
  border-radius: 8px;
  overflow: hidden;
  table-layout: fixed;
}
.transcript-table th {
  background: var(--epam-red);
  color: white;
  text-align: left;
  padding: 10px 12px;
  font-family: Georgia, serif;
  font-size: 14px;
  border-right: 1px solid rgba(255,255,255,0.2);
}
.transcript-table th:last-child { border-right: none; }
.transcript-table td {
  vertical-align: top;
  padding: 10px 12px;
  border-top: 1px solid var(--gray-300);
  border-right: 1px solid var(--gray-300);
  word-wrap: break-word;
  line-height: 1.45;
}
.transcript-table td:last-child { border-right: none; }
.transcript-table tr:nth-child(even) td { background: var(--gray-100); }
.speaker {
  font-weight: bold;
  color: var(--gray-700);
  font-size: 12px;
  display: block;
  margin-bottom: 3px;
}
.timestamp {
  color: var(--gray-500);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  margin-left: 6px;
  font-weight: normal;
}
.col-gold th { background: var(--gray-700); }
.col-empty {
  color: var(--gray-300);
  font-style: italic;
}

.legend {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
  font-size: 12px;
  color: var(--gray-700);
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.legend-dot {
  display: inline-block;
  width: 12px; height: 12px;
  border-radius: 50%;
}
"""


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def wer_class(wer: float) -> str:
    if wer < 0.10:
        return "wer-good"
    if wer < 0.30:
        return "wer-amber"
    return "wer-bad"


def fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def render_html(
    audio_name: str,
    engines: list[tuple[str, list[dict], dict]],  # (engine_name, aligned, score)
    gold: list[dict],
    output_path: Path,
) -> None:
    """Build the side-by-side HTML report.

    The transcript table is laid out with one row per N turns, showing
    gold turn numbers down the left column. For engines, we render
    their segments in chronological order and let CSS visually align
    columns by row index (NOT by content matching — that's why each
    column header notes "in chronological order"). The reader scans
    horizontally to compare what each engine emitted around the same
    point in the audio.
    """
    n_max = max(len(gold), *(len(a) for _, a, _ in engines))

    # Stats panel
    stat_cards = ['''
<div class="stat-card">
  <h3>📋 Gold reference</h3>
  <div class="stat-row"><span class="stat-label">Source</span><span class="stat-value">{name}</span></div>
  <div class="stat-row"><span class="stat-label">Turns</span><span class="stat-value">{n}</span></div>
  <div class="stat-row"><span class="stat-label">Words</span><span class="stat-value">{w}</span></div>
</div>
'''.format(
        name=html.escape("court_hearing_129.jsonl"),
        n=len(gold),
        w=sum(len(str(t.get("text", "")).split()) for t in gold),
    )]

    for engine_name, aligned, sc in engines:
        wer_cls = wer_class(sc["wer"])
        cer_str = fmt_pct(sc["cer"]) if sc.get("cer") is not None else "—"
        speakers = len({s.get("speaker_id") for s in aligned if s.get("speaker_id")})
        stat_cards.append(f'''
<div class="stat-card">
  <h3>🎤 {html.escape(engine_name)}</h3>
  <div class="stat-row"><span class="stat-label">WER</span><span class="stat-value {wer_cls}">{fmt_pct(sc["wer"])}</span></div>
  <div class="stat-row"><span class="stat-label">CER</span><span class="stat-value {wer_cls}">{cer_str}</span></div>
  <div class="stat-row"><span class="stat-label">Substitutions</span><span class="stat-value">{sc["substitutions"]}</span></div>
  <div class="stat-row"><span class="stat-label">Deletions</span><span class="stat-value">{sc["deletions"]}</span></div>
  <div class="stat-row"><span class="stat-label">Insertions</span><span class="stat-value">{sc["insertions"]}</span></div>
  <div class="stat-row"><span class="stat-label">Segments</span><span class="stat-value">{len(aligned)}</span></div>
  <div class="stat-row"><span class="stat-label">Speakers detected</span><span class="stat-value">{speakers}</span></div>
</div>
''')

    legend = '''
<div class="legend">
  <span class="legend-item"><span class="legend-dot" style="background: var(--green);"></span>WER &lt; 10% — good</span>
  <span class="legend-item"><span class="legend-dot" style="background: var(--amber);"></span>10–30% — review</span>
  <span class="legend-item"><span class="legend-dot" style="background: var(--red);"></span>&gt; 30% — poor</span>
</div>
'''

    # Transcript table header
    cols_html = '<col style="width: 28%"><col style="width: 36%"><col style="width: 36%">'
    if len(engines) == 1:
        cols_html = '<col style="width: 40%"><col style="width: 60%">'
    elif len(engines) >= 3:
        cols_html = '<col style="width: 22%">' + '<col style="width: 26%">' * len(engines)

    th_engines = "".join(
        f'<th>{html.escape(name)}<br><span style="font-weight: normal; font-size: 11px;">в хронологическом порядке</span></th>'
        for name, _, _ in engines
    )

    rows: list[str] = []
    for i in range(n_max):
        gold_cell = ""
        if i < len(gold):
            t = gold[i]
            sp = html.escape(str(t.get("speaker", "")))
            txt = html.escape(str(t.get("text", "")).strip())
            gold_cell = (
                f'<span class="speaker">{sp} <span class="timestamp">#{t.get("turn", i+1)}</span></span>'
                f'{txt}'
            )
        else:
            gold_cell = '<span class="col-empty">—</span>'

        engine_cells = []
        for _, aligned, _ in engines:
            if i < len(aligned):
                seg = aligned[i]
                sp = html.escape(
                    str(seg.get("speaker_name") or seg.get("speaker_id", ""))
                )
                ts = fmt_ts(seg.get("start"))
                txt = html.escape(str(seg.get("text", "")).strip())
                engine_cells.append(
                    f'<td><span class="speaker">{sp} '
                    f'<span class="timestamp">{ts}</span></span>{txt}</td>'
                )
            else:
                engine_cells.append('<td><span class="col-empty">—</span></td>')

        rows.append(
            f'<tr><td>{gold_cell}</td>{"".join(engine_cells)}</tr>'
        )

    html_doc = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>VERITAS — ASR engine comparison</title>
<style>{CSS}</style>
</head>
<body>
<h1>Сравнение ASR-движков</h1>
<div class="subtitle">
  Аудио: <strong>{html.escape(audio_name)}</strong> &middot;
  Эталон: <strong>{html.escape(DEFAULT_GOLD.name)}</strong> &middot;
  Сгенерировано: {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div>

<div class="stats">
{"".join(stat_cards)}
</div>

{legend}

<table class="transcript-table">
<colgroup>{cols_html}</colgroup>
<thead>
<tr class="col-gold">
<th>Эталон<br><span style="font-weight: normal; font-size: 11px;">по номерам реплик</span></th>
{th_engines}
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>

</body>
</html>
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    logger.info("HTML written: %s", output_path)


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--audio", type=Path, default=DEFAULT_AUDIO,
        help=f"Audio path (default: {DEFAULT_AUDIO.name})",
    )
    ap.add_argument(
        "--gold", type=Path, default=DEFAULT_GOLD,
        help=f"Gold JSONL path (default: {DEFAULT_GOLD.name})",
    )
    ap.add_argument(
        "--engines", nargs="+", default=["gigaam", "whisperx"],
        choices=["gigaam", "hf-whisper", "whisper", "whisperx"],
        help="Engines to compare (default: gigaam whisperx)",
    )
    ap.add_argument(
        "--skip-runs", action="store_true",
        help=(
            "Skip the pipeline runs and reuse the most recent benchmarks/data/"
            "compare_<engine>_*/aligned.json for each engine. Use after a "
            "previous successful comparison run when you only want to "
            "rebuild the HTML."
        ),
    )
    ap.add_argument(
        "--reuse-existing", action="store_true",
        help=(
            "Per-engine resume: for each engine, if there is already a "
            "compare_<engine>_*/aligned.json in benchmarks/data/, reuse "
            "it; otherwise run the pipeline. Useful to recover from "
            "partial failures (e.g. one engine completed before a missing "
            "scoring dependency killed the run)."
        ),
    )
    args = ap.parse_args()

    if not args.audio.exists():
        logger.error("Audio not found: %s", args.audio)
        return 5
    if not args.gold.exists():
        logger.error("Gold not found: %s", args.gold)
        return 6

    gold = load_gold_jsonl(args.gold)
    gold_text = turns_to_text(gold)

    engines_data: list[tuple[str, list[dict], dict]] = []
    for engine in args.engines:
        prefix = f"compare_{engine.replace('-', '_')}"
        existing = latest_run_dir(prefix)
        if args.skip_runs:
            if existing is None:
                logger.error(
                    "No prior run found for engine=%s. Drop --skip-runs to run it.",
                    engine,
                )
                return 7
            run_dir = existing
            logger.info("Reusing %s for engine=%s", run_dir.name, engine)
        elif args.reuse_existing and existing is not None:
            run_dir = existing
            logger.info(
                "Reusing existing %s for engine=%s (--reuse-existing)",
                run_dir.name, engine,
            )
        else:
            run_dir = run_pipeline(args.audio, engine, prefix)

        aligned = load_aligned_json(run_dir / "aligned.json")
        engine_text = turns_to_text(aligned)
        sc = score_engine(gold_text, engine_text)
        logger.info(
            "engine=%s WER=%.2f%% CER=%s sub=%d del=%d ins=%d",
            engine, sc["wer"] * 100,
            f"{sc['cer'] * 100:.2f}%" if sc.get("cer") is not None else "—",
            sc["substitutions"], sc["deletions"], sc["insertions"],
        )
        engines_data.append((engine, aligned, sc))

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_html = REPO_ROOT / "benchmarks" / "reports" / f"asr_compare_{stamp}.html"
    out_json = REPO_ROOT / "benchmarks" / "reports" / f"asr_compare_{stamp}.json"

    render_html(args.audio.name, engines_data, gold, out_html)

    summary = {
        "audio": str(args.audio),
        "gold": str(args.gold),
        "generated_at": stamp,
        "engines": [
            {
                "name": name,
                "stats": sc,
                "n_segments": len(aligned),
            }
            for name, aligned, sc in engines_data
        ],
    }
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("JSON summary: %s", out_json)
    logger.info("HTML report:  %s", out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
