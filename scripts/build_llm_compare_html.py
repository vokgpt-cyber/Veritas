"""Render a side-by-side HTML view of the 4 LLM-compare protocol outputs.

Input:
    scripts/llm_compare/{stem}__{slug}.protocol.json   (4 files)

Output:
    benchmarks/reports/llm_compare.html

Style matches benchmarks/scripts/score_punctuation.py: EPAM red (#B2001F) for
top-level H1, Georgia for headings, Segoe UI for body, bordered tables, neutral
greys for chrome. Layout is two-column per audio (Gemma left, T-Pro right).
Transcript is omitted; this is a summary-quality diff.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "scripts" / "llm_compare"
OUT_HTML = REPO_ROOT / "benchmarks" / "reports" / "llm_compare.html"

# (audio_stem, model_label, model_slug, model_family)
RUNS = [
    (
        "2026-04-20_Court_hearing_129_01",
        "Court hearing 129_01",
        [
            ("Gemma 4 26B", "gemma4_26b"),
            ("T-Pro 2.0 Q4_K_M", "t-tech_T-pro-it-2.0_q4_K_M"),
        ],
    ),
    (
        "2026-04-20_Админ_13-04-2026_без_мусора_в_начале",
        "Админ 13-04-2026 без мусора в начале",
        [
            ("Gemma 4 26B", "gemma4_26b"),
            ("T-Pro 2.0 Q4_K_M", "t-tech_T-pro-it-2.0_q4_K_M"),
        ],
    ),
]


def load_protocol(stem: str, model_slug: str) -> dict[str, Any]:
    path = IN_DIR / f"{stem}__{model_slug}.protocol.json"
    if not path.exists():
        raise SystemExit(f"missing protocol: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def esc(s: str | None) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def detect_duplicates(items: list[str]) -> list[tuple[str, int]]:
    """Return items that appear more than once, with their counts."""
    if not items:
        return []
    counts = Counter(items)
    return [(item, c) for item, c in counts.items() if c > 1]


def _summary_twin(text: str) -> bool:
    """Heuristic: detect if a summary is two copies of the same paragraph.

    T-Pro has a loop bug where the whole summary is emitted twice verbatim.
    Compare the first half to the second half; if normalised equality -> twin.
    """
    if not text or len(text) < 100:
        return False
    clean = re.sub(r"\s+", " ", text).strip()
    n = len(clean)
    mid = n // 2
    # search for natural split near the midpoint where two identical tails start
    left = clean[:mid].strip()
    right = clean[mid:].strip()
    # also try finding "На встрече" or "В ходе" repeat signatures
    first200 = clean[:200]
    if first200 in clean[200:]:
        return True
    return left[:150] == right[:150]


def render_list(items: list[str], empty_text: str = "(пусто)") -> str:
    if not items:
        return f"<div class='empty'>{esc(empty_text)}</div>"
    lis = "".join(f"<li>{esc(x)}</li>" for x in items)
    return f"<ul>{lis}</ul>"


def render_decisions(decisions: list[dict]) -> str:
    if not decisions:
        return "<div class='empty'>(пусто)</div>"
    items = []
    for d in decisions:
        text = d.get("text", "")
        resp = d.get("responsible")
        tail = f" <span class='meta-inline'>— {esc(resp)}</span>" if resp else ""
        items.append(f"<li>{esc(text)}{tail}</li>")
    return f"<ul>{''.join(items)}</ul>"


def render_tasks(tasks: list[dict]) -> str:
    if not tasks:
        return "<div class='empty'>(пусто)</div>"
    items = []
    for t in tasks:
        text = t.get("text", "")
        assignee = t.get("assignee")
        deadline = t.get("deadline")
        bits = []
        if assignee:
            bits.append(f"👤 {esc(assignee)}")
        if deadline:
            bits.append(f"📅 {esc(deadline)}")
        tail = f" <span class='meta-inline'>— {' · '.join(bits)}</span>" if bits else ""
        items.append(f"<li>{esc(text)}{tail}</li>")
    return f"<ul>{''.join(items)}</ul>"


def render_participants(participants: list[dict]) -> str:
    if not participants:
        return "<div class='empty'>(пусто)</div>"
    # sort descending by share
    parts = sorted(participants, key=lambda p: -p.get("speaking_share", 0))
    rows = []
    for p in parts:
        name = p.get("speaker_name") or p.get("speaker_id") or "?"
        share = p.get("speaking_share", 0)
        secs = p.get("speaking_time", 0)
        mins = int(secs // 60)
        rem = int(secs % 60)
        bar_w = max(1, min(100, int(round(share))))
        rows.append(
            "<tr>"
            f"<td class='pname'>{esc(name)}</td>"
            f"<td class='pbar'><div class='bar' style='width:{bar_w}%'></div></td>"
            f"<td class='pnum'>{share:.1f}%</td>"
            f"<td class='pnum'>{mins}m {rem:02d}s</td>"
            "</tr>"
        )
    return "<table class='participants'><tbody>" + "".join(rows) + "</tbody></table>"


def count_badge(n: int, dup_count: int = 0) -> str:
    if dup_count > 0:
        return (
            f"<span class='count'>{n}</span>"
            f"<span class='warn' title='{dup_count} повторений'>⚠ {dup_count}×</span>"
        )
    return f"<span class='count'>{n}</span>"


def render_column(model_label: str, model_slug: str, stem: str, protocol: dict) -> str:
    summary = protocol.get("summary", "") or ""
    topics = protocol.get("key_topics", []) or []
    decisions = protocol.get("decisions", []) or []
    tasks = protocol.get("tasks", []) or []
    questions = protocol.get("open_questions", []) or []
    participants = protocol.get("participants", []) or []
    topic = protocol.get("topic", "") or ""

    topic_titles = [t.get("title", "") for t in topics]
    dup_topics = detect_duplicates(topic_titles)
    dup_decisions = detect_duplicates([d.get("text", "") for d in decisions])
    dup_tasks = detect_duplicates([t.get("text", "") for t in tasks])
    dup_questions = detect_duplicates(questions)
    summary_twin = _summary_twin(summary)

    warn_bits = []
    if summary_twin:
        warn_bits.append("краткое содержание дублируется")
    if dup_topics:
        warn_bits.append(f"{len(dup_topics)} дубл. тем")
    if dup_decisions:
        warn_bits.append(f"{len(dup_decisions)} дубл. решений")
    if dup_tasks:
        warn_bits.append(f"{len(dup_tasks)} дубл. задач")

    if warn_bits:
        warn_banner = (
            "<div class='banner warn-banner'>⚠ "
            + " · ".join(esc(w) for w in warn_bits)
            + "</div>"
        )
    else:
        warn_banner = "<div class='banner ok-banner'>✓ без дублей</div>"

    summary_html = (
        f"<p class='summary twin'>{esc(summary)}</p>"
        if summary_twin
        else f"<p class='summary'>{esc(summary)}</p>"
    )

    topic_items = [t.get("title") or t.get("content", "") for t in topics]

    stats_row = (
        "<div class='statsrow'>"
        f"<div class='stat'><span class='lbl'>Тем</span>{count_badge(len(topics), len(dup_topics))}</div>"
        f"<div class='stat'><span class='lbl'>Решений</span>{count_badge(len(decisions), len(dup_decisions))}</div>"
        f"<div class='stat'><span class='lbl'>Задач</span>{count_badge(len(tasks), len(dup_tasks))}</div>"
        f"<div class='stat'><span class='lbl'>Вопросов</span>{count_badge(len(questions), len(dup_questions))}</div>"
        f"<div class='stat'><span class='lbl'>Уч.</span>{count_badge(len(participants))}</div>"
        f"<div class='stat'><span class='lbl'>Σ сумм.</span>{count_badge(len(summary))}</div>"
        "</div>"
    )

    return f"""
<div class='col col-{esc(model_slug)}'>
  <div class='colhead'>
    <div class='model'>{esc(model_label)}</div>
    {warn_banner}
  </div>
  <div class='topic'><b>Тема:</b> {esc(topic)}</div>
  {stats_row}
  <h3>Краткое содержание</h3>
  {summary_html}
  <h3>Ключевые темы <span class='count'>{len(topics)}</span></h3>
  {render_list(topic_items)}
  <h3>Решения <span class='count'>{len(decisions)}</span></h3>
  {render_decisions(decisions)}
  <h3>Задачи <span class='count'>{len(tasks)}</span></h3>
  {render_tasks(tasks)}
  <h3>Открытые вопросы <span class='count'>{len(questions)}</span></h3>
  {render_list(questions)}
  <h3>Участники <span class='count'>{len(participants)}</span></h3>
  {render_participants(participants)}
</div>
"""


def render_run(stem: str, label: str, models: list[tuple[str, str]]) -> str:
    cols = []
    for model_label, model_slug in models:
        proto = load_protocol(stem, model_slug)
        cols.append(render_column(model_label, model_slug, stem, proto))
    return f"""
<section class='run'>
  <h2>{esc(label)}</h2>
  <div class='grid'>{"".join(cols)}</div>
</section>
"""


CSS = """
:root {
  --red: #B2001F;
  --ink: #1f1f1f;
  --mute: #555;
  --line: #ddd;
  --bg: #fafafa;
  --card: #fff;
  --warn-bg: #fff3cd;
  --warn-border: #f0c36d;
  --warn-ink: #7a5c00;
  --ok-bg: #e8f5e9;
  --ok-border: #a5d6a7;
  --ok-ink: #2e7d32;
  --bar: #B2001F;
  --bar-bg: #f4e6e8;
  --chip: #f0f0f0;
}
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0;
  padding: 18px 26px 40px 26px; color: var(--ink); background: var(--bg); }
h1 { color: var(--red); font-family: Georgia, serif; margin: 0 0 2px 0; font-size: 24px; }
h2 { font-family: Georgia, serif; margin: 26px 0 10px 0; color: var(--ink);
     font-size: 18px; border-bottom: 2px solid var(--red); padding-bottom: 4px; }
h3 { font-family: Georgia, serif; margin: 14px 0 6px 0; color: var(--ink);
     font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; }
.meta { color: var(--mute); font-size: 13px; margin-bottom: 16px; line-height: 1.5; }
.meta code { background: var(--chip); padding: 1px 4px; border-radius: 3px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.col { background: var(--card); border: 1px solid var(--line); border-radius: 4px;
  padding: 12px 16px 16px 16px; }
.colhead { display: flex; flex-wrap: wrap; justify-content: space-between;
  align-items: center; gap: 10px; margin-bottom: 8px;
  border-bottom: 1px solid var(--line); padding-bottom: 6px; }
.model { font-family: Georgia, serif; font-size: 15px; font-weight: bold;
  color: var(--red); }
.banner { font-size: 12px; padding: 3px 10px; border-radius: 3px;
  border: 1px solid var(--line); }
.warn-banner { background: var(--warn-bg); border-color: var(--warn-border);
  color: var(--warn-ink); }
.ok-banner { background: var(--ok-bg); border-color: var(--ok-border);
  color: var(--ok-ink); }
.topic { font-size: 13px; margin: 6px 0 10px 0; color: var(--ink); }
.topic b { color: var(--red); }
.statsrow { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 4px; }
.stat { font-size: 12px; background: var(--chip); border-radius: 3px;
  padding: 3px 8px; display: flex; align-items: center; gap: 6px; }
.stat .lbl { color: var(--mute); text-transform: uppercase;
  letter-spacing: 0.04em; font-size: 11px; }
.count { font-variant-numeric: tabular-nums; font-family: Consolas, monospace;
  font-weight: bold; color: var(--ink); }
.warn { color: var(--warn-ink); font-family: Consolas, monospace;
  font-weight: bold; font-size: 11px; }
.summary { font-size: 13px; line-height: 1.55; margin: 4px 0 8px 0;
  padding: 8px 10px; background: #fcfcfc; border-left: 3px solid var(--red);
  border-radius: 2px; }
.summary.twin { background: var(--warn-bg); border-left-color: var(--warn-border); }
ul { margin: 4px 0 8px 22px; padding: 0; font-size: 13px; line-height: 1.5; }
li { margin: 3px 0; }
.empty { font-size: 12px; color: var(--mute); font-style: italic;
  padding: 4px 0 6px 0; }
.meta-inline { color: var(--mute); font-size: 12px; }
table.participants { width: 100%; border-collapse: collapse; font-size: 12px;
  margin-top: 4px; }
table.participants td { border: none; padding: 2px 6px; }
td.pname { font-family: Consolas, monospace; width: 120px; color: var(--ink); }
td.pbar { width: 60%; }
td.pnum { text-align: right; font-variant-numeric: tabular-nums;
  font-family: Consolas, monospace; color: var(--mute); width: 60px; }
.bar { height: 8px; background: var(--bar); border-radius: 2px; }
.bar-outer { width: 100%; background: var(--bar-bg); height: 8px;
  border-radius: 2px; }
.legend { background: var(--card); border: 1px solid var(--line); padding: 8px 14px;
  border-radius: 3px; margin-top: 4px; font-size: 12px; color: var(--mute); }
.legend b { color: var(--ink); }
@media (max-width: 1100px) {
  .grid { grid-template-columns: 1fr; }
}
"""


def build_html() -> str:
    sections = "".join(
        render_run(stem, label, models) for stem, label, models in RUNS
    )
    meta = """
<div class='meta'>
Сравнение двух LLM-бэкендов суммаризации: <b>Gemma 4 26B (MoE, Ollama)</b> против
<b>T-Pro 2.0 Q4_K_M (T-Bank)</b>, при идентичных транскрипциях GigaAM v3 + pyannote 4.0.
Оба прогона использовали один промпт (7 секций, английские заголовки, русский контент),
температуру <code>0.2</code>, top-p <code>0.85</code>, плюс verification-pass
(второй вызов LLM, фильтрующий утверждения без опоры на транскрипт).
Транскрипты не показаны (идентичны у обоих бэкендов) — виден только вывод суммаризатора.
</div>
<div class='legend'>
  <b>Как читать:</b>
  <span class='banner warn-banner'>⚠ …</span> — обнаружены дубликаты/повторы в выводе.
  <span class='banner ok-banner'>✓ без дублей</span> — вывод чистый.
  Цифры рядом с заголовками секций — количество элементов. «Σ сумм.» — длина
  summary в символах.
</div>
"""
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>LLM A/B — Gemma 4 vs T-Pro 2.0 (протоколы)</title>
<style>{CSS}</style>
</head><body>
<h1>LLM A/B — Gemma 4 26B vs T-Pro 2.0 Q4_K_M</h1>
{meta}
{sections}
</body></html>
"""


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
