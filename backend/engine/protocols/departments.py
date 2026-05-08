"""EPAM internal department taxonomy for administrative meetings.

Order matches the HTML administrative meeting agenda used internally at
EPAM (reference: user-provided presentation 2026-04-21). This order is
the display order in the protocol DOCX: HR first, then Finance, etc.

User-specified addition 2026-04-21: AHO (Административно-хозяйственный
отдел) added after IAO (Информационно-аналитический отдел). AHO has no
tasks for now but must appear in the DOCX section list — the formatter
renders "Нет задач" for empty departments so readers know nothing was
missed for that department.

If a task or decision can't be matched to any department, it goes into
the OTHER bucket (Прочее) at the end.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Department:
    """A department entry: slug (stable key), display name (Cyrillic),
    short code (abbreviation shown in UI). Ordered by display_order."""

    slug: str
    display_name: str
    short_code: str
    display_order: int
    # Aliases the LLM might emit — normalised to this department at parse time.
    # Include transliterated variants because Gemma 4 sometimes translits.
    aliases: tuple[str, ...] = ()


# Canonical department list. display_order is the rank in the DOCX.
# Keep display_order gaps (10, 20, 30...) so new departments can be
# inserted without renumbering everything.
DEPARTMENTS: tuple[Department, ...] = (
    Department(
        slug="hr",
        display_name="HR-отдел",
        short_code="HR",
        display_order=10,
        aliases=(
            "HR", "HR отдел", "HR-отдел", "Отдел HR",
            "Кадры", "Отдел кадров", "Кадровый отдел",
            "Human Resources", "HR department",
        ),
    ),
    Department(
        slug="finance",
        display_name="Финансовый отдел",
        short_code="Финансы",
        display_order=20,
        aliases=(
            "Финансы", "Финансовый отдел", "Финансовое",
            "Финансовое управление", "Бухгалтерия",
            "Finance", "Finance department", "Financial department",
        ),
    ),
    Department(
        slug="it",
        display_name="Отдел ИТ",
        short_code="ИТ",
        display_order=30,
        aliases=(
            "ИТ", "IT", "Отдел ИТ", "Отдел IT", "ИТ-отдел",
            "IT department", "Information technology",
            "Технический отдел",
        ),
    ),
    Department(
        slug="pr_marketing",
        display_name="PR и маркетинг",
        short_code="PR",
        display_order=40,
        aliases=(
            "PR", "PR и маркетинг", "Маркетинг", "PR-отдел",
            "PR department", "Marketing", "Marketing department",
            "PR & Marketing",
        ),
    ),
    Department(
        slug="iao",
        display_name="ИАО (Информационно-аналитический отдел)",
        short_code="ИАО",
        display_order=70,
        aliases=(
            "ИАО", "IAO", "Информационно-аналитический отдел",
            "Информационно-аналитический",
            "Information and Analytical Department",
        ),
    ),
    # User-added 2026-04-21: АХО after ИАО, typically no tasks but must appear.
    Department(
        slug="aho",
        display_name="АХО (Административно-хозяйственный отдел)",
        short_code="АХО",
        display_order=75,
        aliases=(
            "АХО", "AHO", "Административно-хозяйственный отдел",
            "Административно-хозяйственный",
            "Administrative and Economic Department",
            "Хозяйственный отдел",
        ),
    ),
    Department(
        slug="other",
        display_name="Прочее",
        short_code="Прочее",
        display_order=99,
        aliases=(
            "Прочее", "Other", "Другое", "Разное",
            "Miscellaneous", "Прочие вопросы",
        ),
    ),
)

# Fast lookup by slug.
DEPARTMENT_BY_SLUG: dict[str, Department] = {d.slug: d for d in DEPARTMENTS}

# Ordered slugs for iteration (matches display_order).
DEPARTMENT_SLUGS_ORDERED: tuple[str, ...] = tuple(
    d.slug for d in sorted(DEPARTMENTS, key=lambda d: d.display_order)
)


def resolve_department_slug(raw: str | None) -> str:
    """Normalise an LLM-provided department string to a canonical slug.

    Matches case-insensitively against slug, short_code, display_name,
    and all aliases. Falls back to 'other' when nothing matches so we
    never silently drop tasks.

    Args:
        raw: Department name as emitted by the LLM, or None.

    Returns:
        Canonical slug (one of DEPARTMENT_SLUGS_ORDERED).
    """
    if not raw:
        return "other"

    needle = raw.strip().lower()
    if not needle:
        return "other"

    # Strip trailing "отдел" / "department" for looser matching.
    stripped = needle
    for suffix in (
        " отдел", " department", " департамент", "-отдел",
    ):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()

    for dept in DEPARTMENTS:
        candidates = {
            dept.slug.lower(),
            dept.short_code.lower(),
            dept.display_name.lower(),
        }
        candidates.update(a.lower() for a in dept.aliases)
        # Match on full string OR stripped-suffix string OR substring.
        if needle in candidates or stripped in candidates:
            return dept.slug
        # Substring: LLM often emits "HR-отдел компании EPAM" etc.
        for cand in candidates:
            if cand and (cand in needle or cand in stripped):
                return dept.slug

    return "other"


def list_departments_for_prompt() -> str:
    """Format the department list for LLM prompts.

    Returns a numbered list matching the DOCX display order, used inside
    the administrative meeting prompt so the LLM knows the canonical
    vocabulary.
    """
    lines = []
    for i, slug in enumerate(DEPARTMENT_SLUGS_ORDERED, start=1):
        dept = DEPARTMENT_BY_SLUG[slug]
        lines.append(f"  {i}. {dept.display_name}")
    return "\n".join(lines)


__all__ = [
    "Department",
    "DEPARTMENTS",
    "DEPARTMENT_BY_SLUG",
    "DEPARTMENT_SLUGS_ORDERED",
    "resolve_department_slug",
    "list_departments_for_prompt",
]
