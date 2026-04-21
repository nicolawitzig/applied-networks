"""
Heuristic pre-coder for account voice types (Table 1 in the paper).

The paper coded every account manually. That's still the gold standard.
These heuristics give a first-pass label so manual coders can correct rather
than start from scratch, and so the pipeline is runnable end-to-end without
coders.

Categories (see Table 1):
  1. official_institutional — the 2 official UZH accounts (seeded)
  2. official_individual    — (none found empirically)
  3. central_institutional  — university-wide service units
  4. central_individual     — spokespeople, central admin staff
  5. decentral_institutional — departments, institutes, labs, centres
  6. decentral_individual   — professors, researchers, postdocs, PhDs, students
  7. former_institutional   — spinoffs (not found empirically)
  8. former_individual      — alumni, former employees
"""
from __future__ import annotations

import re

INSTITUTIONAL_KEYWORDS = [
    "department", "institute", "lab", "laboratory", "group", "center", "centre",
    "faculty", "school of", "division", "chair", "clinic", "hospital",
    "office", "association", "society", "programme", "program",
    "departement", "institut", "fakultät", "lehrstuhl", "zentrum",
]

CENTRAL_UNIT_KEYWORDS = [
    "alumni", "equal opportunity", "gender equality", "international office",
    "career services", "graduate campus", "graduate career",
    "media relations", "press office", "communications office",
]

FORMER_KEYWORDS = [
    "former", "ex-", "alumnus", "alumna", "alum", "previously at",
    "ehemalig", "ehemalige",
]

CAREER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("professor", re.compile(r"\b(prof(essor)?|prof\.)\b", re.I)),
    ("postdoc", re.compile(r"\b(post[- ]?doc|postdoctoral)\b", re.I)),
    ("phd", re.compile(r"\b(ph\.?d\.?|doctoral candidate|doktorand)\b", re.I)),
    ("researcher", re.compile(r"\b(researcher|scientist|research(er)? associate|wissenschaftlich)\b", re.I)),
    ("student", re.compile(r"\b(student|studierend|studying at)\b", re.I)),
]


def _has_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def classify(description: str, handle: str, display_name: str) -> dict:
    """
    Return {voice_type, subtype, is_institutional} as a first-pass guess.
    `voice_type` is one of the 8 categories above (or 'unknown').
    """
    desc = description or ""
    name = display_name or ""
    blob = f"{desc} {name}".strip()

    is_institutional = _has_any(blob, INSTITUTIONAL_KEYWORDS)
    is_former = _has_any(blob, FORMER_KEYWORDS)
    is_central_unit = _has_any(blob, CENTRAL_UNIT_KEYWORDS)

    subtype = None
    if not is_institutional:
        for label, pat in CAREER_PATTERNS:
            if pat.search(blob):
                subtype = label
                break

    if is_institutional and is_central_unit:
        return {"voice_type": "central_institutional", "subtype": "administrative_body", "is_institutional": True}
    if is_institutional and is_former:
        return {"voice_type": "former_institutional", "subtype": "unit", "is_institutional": True}
    if is_institutional:
        return {"voice_type": "decentral_institutional", "subtype": "department_or_lab", "is_institutional": True}

    if is_former:
        return {"voice_type": "former_individual", "subtype": subtype or "employee_or_alumni", "is_institutional": False}

    if subtype:
        return {"voice_type": "decentral_individual", "subtype": subtype, "is_institutional": False}

    return {"voice_type": "unknown", "subtype": None, "is_institutional": False}
