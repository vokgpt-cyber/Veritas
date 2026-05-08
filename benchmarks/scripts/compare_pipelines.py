"""A/B comparison harness: topic-segmented vs single-pass admin pipeline.

Sprint 2026-05-04 (task #47). Decision-driven evaluation of the
un-engineering sprint:

* Pipeline A: topic-segmented admin path (map-reduce flag on).
* Pipeline B: default single-pass path with direct prompt, implicit
  decisions, per-item confidence, and speaker resolution post-pass.

Reads existing aligned.json transcripts from the meetings working
directory or the backups folder, runs BOTH pipelines on the same
input, generates a side-by-side HTML report. Decision goes to whichever
produces more useful output to the human eye.

Usage::

    python benchmarks/scripts/compare_pipelines.py \\
        --input data/meetings/<job_id>/aligned.json \\
        --context "Участники совещания: Р.Жавнер, А.Мартынова, ..." \\
        --output benchmarks/reports/compare_admin.html

If --context is omitted, pulls it from the job's saved context if
available. Multiple --input flags compare across several archived
meetings in one run (one HTML page per meeting, indexed in a TOC).

The script does NOT re-run ASR or diarization — it only re-runs the
summarization layer on the already-aligned transcript. This makes
iteration cheap (LLM-only, no GPU loading of ASR/diarization).
"""

from __future__ import annotations

import argparse
import asyncio
import html as html_lib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from project root via `python benchmarks/scripts/...`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.config import AppConfig  # noqa: E402
from backend.app.models import AlignedSegment, MeetingType  # noqa: E402
from backend.engine.summarization import SummarizationEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compare_pipelines")


def load_aligned(path: Path) -> list[AlignedSegment]:
    """Load aligned.json into AlignedSegment objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of segments in {path}, got {type(raw)}")
    return [
        AlignedSegment(**item) if isinstance(item, dict) else item
        for item in raw
    ]


def fetch_meeting_context(aligned_path: Path) -> str:
    """Best-effort fetch of the meeting context for an archived job.

    Looks for a sibling status.json or similar file in the same
    directory and pulls the `context` field. Empty string if absent.
    """
    job_dir = aligned_path.parent
    for candidate in ("status.json", "job.json"):
        p = job_dir / candidate
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    ctx = data.get("context")
                    if isinstance(ctx, str):
                        return ctx
            except Exception:  # noqa: BLE001
                continue
    return ""


async def run_pipeline(
    *,
    label: str,
    aligned: list[AlignedSegment],
    meeting_context: str,
    use_topic_segmented: bool,
) -> dict[str, Any]:
    """Run the summarization pipeline once with the given config flag.

    Returns a dict with the parsed protocol payload (as JSON-able dict)
    plus timing info. Loads + unloads the engine so each run is
    independent.
    """
    config = AppConfig.load_from_yaml(_PROJECT_ROOT / "config" / "settings.yaml")
    config.summarization.use_topic_segmented_admin = use_topic_segmented

    engine = SummarizationEngine(config)
    engine.load()
    t0 = time.time()
    try:
        result = await engine.process(
            aligned,
            participants=None,
            language="ru",
            meeting_context=meeting_context,
            meeting_type=MeetingType.ADMINISTRATIVE,
        )
    finally:
        engine.unload()
    elapsed = time.time() - t0

    payload = getattr(result, "payload", None)
    payload_dict: dict[str, Any] = {}
    if payload is not None and hasattr(payload, "model_dump"):
        payload_dict = payload.model_dump(mode="json")
    elif payload is not None:
        payload_dict = dict(payload)

    return {
        "label": label,
        "elapsed_s": elapsed,
        "payload": payload_dict,
    }


def render_items_section(title: str, items: list[dict[str, Any]]) -> str:
    """Render a list of decisions/tasks/etc. as HTML."""
    if not items:
        return f"<h4>{html_lib.escape(title)}</h4><p class='empty'>—</p>"
    lines = [f"<h4>{html_lib.escape(title)}</h4>", "<ul>"]
    for it in items:
        text = html_lib.escape(str(it.get("text", "")))
        confidence = (it.get("confidence") or "medium").lower()
        speaker = html_lib.escape(str(it.get("speaker") or ""))
        owner = html_lib.escape(str(it.get("owner") or ""))
        deadline = html_lib.escape(str(it.get("deadline") or ""))
        evidence = html_lib.escape(str(it.get("evidence") or ""))

        cls = "low-conf" if confidence == "low" else ""
        meta_bits: list[str] = []
        if speaker:
            meta_bits.append(f"<em>{speaker}</em>")
        if owner:
            meta_bits.append(f"исполнитель: <strong>{owner}</strong>")
        if deadline:
            meta_bits.append(f"срок: <strong>{deadline}</strong>")
        if evidence:
            meta_bits.append(f"<small>«{evidence}»</small>")
        if confidence == "low":
            meta_bits.append("<span class='conf-tag'>?</span>")
        meta = " · ".join(meta_bits)

        lines.append(
            f"<li class='{cls}'>{text}"
            + (f"<div class='meta'>{meta}</div>" if meta else "")
            + "</li>"
        )
    lines.append("</ul>")
    return "\n".join(lines)


def render_pipeline_column(run: dict[str, Any]) -> str:
    """Render one pipeline result as a single HTML column."""
    payload = run["payload"]
    summary = html_lib.escape(payload.get("summary") or "")
    goal = html_lib.escape(payload.get("meeting_goal") or "")
    participants = payload.get("participants") or []
    topics = payload.get("topics") or []
    decisions = payload.get("decisions") or []
    tasks = payload.get("tasks") or []
    open_questions = payload.get("open_questions") or []
    risks = payload.get("risks") or []
    topic_summaries = payload.get("topic_summaries") or []

    parts = [
        f"<h2>{html_lib.escape(run['label'])}</h2>",
        f"<div class='timing'>Время: {run['elapsed_s']:.1f} с</div>",
    ]
    if goal:
        parts.append(f"<h4>Цель встречи</h4><p>{goal}</p>")
    if summary:
        parts.append(f"<h4>Краткое содержание</h4><p>{summary}</p>")
    if participants:
        parts.append(
            "<h4>Участники</h4><p>"
            + ", ".join(html_lib.escape(str(p)) for p in participants)
            + "</p>"
        )
    if topics:
        parts.append(
            "<h4>Темы</h4><ul>"
            + "".join(f"<li>{html_lib.escape(str(t))}</li>" for t in topics)
            + "</ul>"
        )

    # Topic-segmented blocks (only present when map-reduce ran).
    if topic_summaries:
        parts.append("<h4>Тематические блоки (map-reduce)</h4>")
        for ts in topic_summaries:
            parts.append(
                f"<div class='topic-block'><h5>{html_lib.escape(str(ts.get('name', '')))}</h5>"
            )
            disc = html_lib.escape(str(ts.get("discussion") or ""))
            if disc:
                parts.append(f"<p><em>{disc}</em></p>")
            parts.append(render_items_section("Решения", ts.get("decisions") or []))
            parts.append(render_items_section("Задачи", ts.get("tasks") or []))
            parts.append(
                render_items_section("Открытые вопросы", ts.get("open_questions") or [])
            )
            parts.append("</div>")

    parts.append(render_items_section("Решения (плоско)", decisions))
    parts.append(render_items_section("Задачи (плоско)", tasks))
    parts.append(render_items_section("Открытые вопросы (плоско)", open_questions))
    parts.append(render_items_section("Риски и блокеры", risks))
    return "\n".join(parts)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; color: #333; }
h1 { color: #B2001F; border-bottom: 2px solid #B2001F; padding-bottom: 8px; }
h2 { color: #B2001F; margin-top: 0; }
h4 { color: #555; margin: 18px 0 6px; font-size: 14px; }
h5 { color: #B2001F; margin: 12px 0 4px; font-size: 13px; }
.layout { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.column { border: 1px solid #ddd; padding: 18px; border-radius: 6px; background: #fff; }
.timing { color: #888; font-size: 12px; margin-bottom: 12px; }
.topic-block { background: #fafafa; padding: 8px 12px; margin: 8px 0; border-left: 3px solid #B2001F; }
ul { padding-left: 22px; margin: 4px 0; }
li { margin-bottom: 6px; line-height: 1.4; }
li.low-conf { background: #fff8e1; padding: 4px 8px; border-radius: 4px; border-left: 3px solid #f39c12; }
.meta { font-size: 11px; color: #666; margin-top: 2px; }
.meta em { color: #555; font-style: normal; font-weight: 500; }
.meta small { color: #999; font-style: italic; }
.conf-tag { background: #f39c12; color: white; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: bold; }
.empty { color: #aaa; font-style: italic; }
.summary-stats { background: #f5f5f5; padding: 12px; border-radius: 4px; margin-bottom: 24px; }
.summary-stats td { padding: 4px 12px; }
"""


def count_items(payload: dict[str, Any]) -> dict[str, int]:
    """Count items per section for the summary statistics row."""
    return {
        "decisions": len(payload.get("decisions") or []),
        "tasks": len(payload.get("tasks") or []),
        "open_questions": len(payload.get("open_questions") or []),
        "risks": len(payload.get("risks") or []),
        "topics": len(payload.get("topics") or []),
        "topic_summaries": len(payload.get("topic_summaries") or []),
    }


def render_summary_stats(old: dict[str, Any], new: dict[str, Any]) -> str:
    o = count_items(old["payload"])
    n = count_items(new["payload"])
    rows = []
    for key, label in [
        ("decisions", "Решения"),
        ("tasks", "Задачи"),
        ("open_questions", "Открытые вопросы"),
        ("risks", "Риски"),
        ("topics", "Темы"),
        ("topic_summaries", "Тематические блоки"),
    ]:
        rows.append(
            f"<tr><td>{html_lib.escape(label)}</td>"
            f"<td>{o[key]}</td><td>{n[key]}</td>"
            f"<td>{n[key] - o[key]:+d}</td></tr>"
        )
    return f"""
<div class='summary-stats'>
  <h3>Сводная статистика</h3>
  <table>
    <tr><th>Section</th><th>Topic-segmented</th><th>Single-pass</th><th>Delta</th></tr>
    {''.join(rows)}
    <tr><td><em>Время прогона</em></td>
        <td>{old['elapsed_s']:.1f} с</td>
        <td>{new['elapsed_s']:.1f} с</td>
        <td>{new['elapsed_s'] - old['elapsed_s']:+.1f} с</td></tr>
  </table>
</div>
"""


def render_html(
    *,
    title: str,
    old_run: dict[str, Any],
    new_run: dict[str, Any],
) -> str:
    summary = render_summary_stats(old_run, new_run)
    return f"""<!DOCTYPE html>
<html lang='ru'>
<head>
  <meta charset='UTF-8'>
  <title>{html_lib.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>A/B сравнение пайплайнов: {html_lib.escape(title)}</h1>
  <p>Left: topic-segmented admin path. Right: default single-pass path
     with direct extraction, confidence labels, and speaker resolution.</p>
  {summary}
  <div class='layout'>
    <div class='column'>{render_pipeline_column(old_run)}</div>
    <div class='column'>{render_pipeline_column(new_run)}</div>
  </div>
</body>
</html>"""


async def main() -> int:
    parser = argparse.ArgumentParser(description="A/B compare admin pipelines")
    parser.add_argument(
        "--input", action="append", required=True,
        help="Path to aligned.json. Repeat for multiple meetings.",
    )
    parser.add_argument(
        "--context", default=None,
        help="Meeting context (overrides any context found in job dir).",
    )
    parser.add_argument(
        "--output", default="benchmarks/reports/compare_admin.html",
        help="Output HTML path. For multiple inputs, suffixed per file.",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    for inp in inputs:
        if not inp.exists():
            logger.error("Input not found: %s", inp)
            return 1

    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for i, inp in enumerate(inputs):
        logger.info("=== Running comparison for %s ===", inp)
        aligned = load_aligned(inp)
        meeting_context = args.context if args.context is not None else fetch_meeting_context(inp)
        if not meeting_context:
            logger.warning(
                "No meeting context found for %s — speaker resolution will skip",
                inp,
            )

        logger.info("Running topic-segmented pipeline...")
        old_run = await run_pipeline(
            label="A: topic-segmented",
            aligned=aligned,
            meeting_context=meeting_context,
            use_topic_segmented=True,
        )
        logger.info("Topic-segmented pipeline done in %.1fs", old_run["elapsed_s"])

        logger.info("Running single-pass pipeline...")
        new_run = await run_pipeline(
            label="B: single-pass + direct prompt",
            aligned=aligned,
            meeting_context=meeting_context,
            use_topic_segmented=False,
        )
        logger.info("Single-pass pipeline done in %.1fs", new_run["elapsed_s"])

        title = inp.parent.name
        if len(inputs) == 1:
            out_path = output_base
        else:
            out_path = output_base.with_stem(f"{output_base.stem}_{i+1}_{title}")

        out_path.write_text(
            render_html(title=title, old_run=old_run, new_run=new_run),
            encoding="utf-8",
        )
        logger.info("Report written: %s", out_path)

        # Also save raw payloads as JSON for downstream analysis.
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {"old": old_run, "new": new_run},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Raw payloads written: %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
