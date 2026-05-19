#!/usr/bin/env python3
"""
Classify Reddit users' posts into the ETH Zurich voice typology (Volk et al. 2024).
Input:  JSON files in results/user_posts/ (one per user, produced by scrape_user_posts.py).
Output: per-user CSV + distribution summary CSV analogous to table2_voice_distribution.csv.

Usage:
  python classify_content.py
  python classify_content.py --input-dir results/user_posts --output-dir results/
  python classify_content.py --model llama3.2 --max-chars 1200 --resume
"""

import argparse
import csv
import json  # used by load_user_data
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Typology constants
# ---------------------------------------------------------------------------

VOICE_TYPES = {  # all valid top-level voice type labels
    "central_institutional",
    "decentral_individual",
    "decentral_institutional",
    "external_individual",  # affiliated with a university other than ETH
    "former_individual",  # formerly affiliated with ETH (alumni, ex-staff)
    "former_institutional",  # formerly an ETH unit
    "unknown",
}

VALID_COMBOS: frozenset = frozenset(
    {  # (voice_type, subtype) pairs from Volk et al.
        ("central_institutional", "administrative_body"),
        ("decentral_individual", "applicant"),
        ("decentral_individual", "phd"),
        ("decentral_individual", "postdoc"),
        ("decentral_individual", "professor"),
        ("decentral_individual", "researcher"),
        ("decentral_individual", "student"),
        ("decentral_institutional", "department_or_lab"),
        ("external_individual", "student"),
        ("external_individual", "phd"),
        ("external_individual", "postdoc"),
        ("external_individual", "professor"),
        ("external_individual", "researcher"),
        ("external_individual", "other"),
        ("former_individual", "professor"),
        ("former_individual", "postdoc"),
        ("former_individual", "phd"),
        ("former_individual", "researcher"),
        ("former_individual", "student"),
        ("former_individual", "employee_or_alumni"),
        ("former_institutional", "unit"),
        ("unknown", ""),
    }
)

# Maps common LLM subtype variations to canonical VALID_COMBOS labels.
_SUBTYPE_ALIASES: Dict[str, str] = {
    "master": "student",
    "masters": "student",
    "master_student": "student",
    "master student": "student",
    "msc": "student",
    "bsc": "student",
    "undergraduate": "student",
    "bachelor": "student",
    "bachelors": "student",
    "doctoral": "phd",
    "doctorate": "phd",
    "phd_student": "phd",
    "phd candidate": "phd",
    "phd_candidate": "phd",
    "doctoral_student": "phd",
    "associate_professor": "professor",
    "assistant_professor": "professor",
    "full_professor": "professor",
    "lecturer": "professor",
    "postdoctoral": "postdoc",
    "post_doc": "postdoc",
    "post-doc": "postdoc",
    "research_associate": "researcher",
    "scientist": "researcher",
    "alumni": "employee_or_alumni",
    "alumnus": "employee_or_alumni",
    "alumna": "employee_or_alumni",
    "graduate": "employee_or_alumni",
    "employee": "employee_or_alumni",
    "lab": "department_or_lab",
    "laboratory": "department_or_lab",
    "institute": "department_or_lab",
    "department": "department_or_lab",
    "research_group": "department_or_lab",
    "centre": "department_or_lab",
    "admin": "administrative_body",
    "administration": "administrative_body",
}

# ---------------------------------------------------------------------------
# Content filtering
# ---------------------------------------------------------------------------

ETHZ_SUBREDDIT = "ethz"  # home subreddit — all posts included unconditionally

_AFFILIATION_RE = re.compile(  # academic-role keywords; gates non-ethz posts/comments
    r"\b(phd|ph\.d|doctorate|doctoral|master|masters|msc|bachelor|bachelors|"
    r"undergraduate|university|postdoc|post[\-\s]doc|professor|researcher|"
    r"thesis|dissertation|semester|faculty|epfl|zhaw|eth\s+z[uü]rich)\b",
    re.IGNORECASE,
)


def _contains_affiliation_keyword(text: str) -> bool:
    """True if text matches any academic-affiliation keyword in _AFFILIATION_RE."""
    return bool(_AFFILIATION_RE.search(text))


def _is_relevant(subreddit: str, *text_parts: str) -> bool:
    """True if post/comment is from r/ethz OR any text_part contains an affiliation keyword."""
    if subreddit.lower() == ETHZ_SUBREDDIT:
        return True
    return _contains_affiliation_keyword(" ".join(text_parts))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class UserData:
    """Raw content from one user JSON file — username + all posts/comments."""

    username: str  # Reddit username, also the JSON filename stem
    posts: List[Dict]  # list of post objects (title, selftext, ...)
    comments: List[Dict]  # list of comment objects (body, ...)


@dataclass
class Classification:
    """LLM-produced voice type classification for a single Reddit user."""

    username: str  # Reddit username
    voice_type: str  # top-level category (VOICE_TYPES)
    subtype: str  # role within voice_type (e.g. "phd", "professor")
    confidence: float  # LLM self-reported confidence [0.0, 1.0]
    n_posts: int  # total posts + comments used as classification evidence
    evidence: str  # verbatim quote from the text that grounds the classification
    reasoning: str  # one-sentence explanation from the LLM


# ---------------------------------------------------------------------------
# Ollama inference config
# ---------------------------------------------------------------------------

MAX_RETRIES = 3  # attempts per user before giving up and returning unknown
RETRY_DELAY = 5  # seconds between retries on transient Ollama errors

EXTRACTION_OPTIONS: Dict = {  # stage 1 — short JSON array; 128 tokens is sufficient
    "temperature": 0,
    "num_predict": 128,
    "num_ctx": 2048,
}

OLLAMA_OPTIONS: Dict = {  # stage 2 — JSON object with 5 fields
    "temperature": 0,
    "num_predict": 256,
    "num_ctx": 2048,
}

# JSON schemas passed as format= to Ollama for grammar-constrained structured output.
EXTRACTION_SCHEMA: Dict = {
    "type": "object",
    "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
    "required": ["statements"],
    "additionalProperties": False,
}

CLASSIFICATION_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "voice_type": {"type": "string"},
        "subtype":    {"type": "string"},
        "confidence": {"type": "number"},
        "evidence":   {"type": "string"},
        "reasoning":  {"type": "string"},
    },
    "required": ["voice_type", "subtype", "confidence", "evidence", "reasoning"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """Extract verbatim first-person statements from Reddit posts/comments
where the user describes their OWN current or past university affiliation or role.

Include ONLY direct first-person claims:
  "I study at ...", "I'm a PhD at ...", "I graduated from ...",
  "I'm doing my master's here", "I work as a researcher at ..."

Exclude ALL of:
- Advice or recommendations to others ("you should go to ETH", "consider EPFL")
- Second-person text ("You're gonna pay more at Cambridge")
- Questions ("which program should I apply to?")
- Preferences or hypotheticals ("maybe I should go to ZHAW", "viellicht doch lieber zhaw")
- Third-party statements — about flatmates, friends, supervisors, colleagues
- General institutional discussion with no personal claim attached
- Joke or non-literal text ("Step 1: apply, Step 2: ???, Step 4: profit")

If no qualifying statement exists, return an empty array.
Return ONLY valid JSON: {"statements": ["<verbatim quote>", ...]}"""

EXTRACTION_TEMPLATE = """Posts and comments by u/{username} in r/ethz:

{text_sample}

Extract all first-person affiliation statements as JSON."""

CLASSIFICATION_SYSTEM = """Classify a Reddit user's university affiliation from the
self-identifying statements provided. ETH Zurich is the HOME institution — it is NOT external.
These statements were pre-extracted from r/ethz posts.

Subreddit default: if a statement mentions studying, doing a PhD/master/bachelor/Basisjahr/
thesis, or working as a researcher WITHOUT naming a specific other university → assume ETH Zurich.

Decision tree (stop at first match — CURRENT affiliation wins):
A) User explicitly names a non-ETH institution as their OWN current role, with no current or
   upcoming ETH connection → external_individual
   Examples: "I study at EPFL" / "I'm a lecturer at ZHAW"
B) User states a CURRENT or UPCOMING ETH affiliation — enrolled, accepted, starting soon —
   even if a prior non-ETH degree is mentioned → decentral_individual | central_institutional | decentral_institutional
   Examples: "I got accepted to ETH MSc" / "starting ETH this fall" / "I'm in my 2nd semester"
             "did my bachelor's at TUM, doing my master's here" → decentral_individual/student
   Note: being accepted to ETH supersedes a prior/concurrent external institution.
B2) User has APPLIED to ETH but has NOT yet confirmed acceptance → decentral_individual/applicant
    Examples: "I applied to ETH's CS MSc" / "I submitted my application"
C) User mentions ETH only in the past (graduated, alumni, left) → former_individual | former_institutional
D) No clear affiliation → unknown

TENSE AND ROLE MATTERS:
- "ZHAW grad here, ten years in tech" → graduated (past) + industry (present) → external_individual/other
- "I study at EPFL" → current student → external_individual/student

external_individual requires POSITIVE evidence of a named non-ETH institution.
"No ETH affiliation visible" alone → use unknown.

Voice types and subtypes:
  decentral_individual:    applicant | student | phd | postdoc | researcher | professor
  central_institutional:   administrative_body
  decentral_institutional: department_or_lab
  former_individual:       student | phd | postdoc | researcher | professor | employee_or_alumni
  former_institutional:    unit
  external_individual:     student | phd | postdoc | researcher | professor | other
  unknown:                 (no subtype — use empty string "")

Confidence: 0.8-1.0 explicit first-person statement, 0.5-0.7 strong implicit signal,
0.2-0.4 weak signal.

Return ONLY valid JSON:
{"voice_type": "...", "subtype": "...", "confidence": 0.0, "evidence": "...", "reasoning": "..."}"""

CLASSIFICATION_TEMPLATE = """Self-identifying statements by u/{username}:

{statements_text}

Classify their university affiliation as JSON."""

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_user_data(path: Path) -> UserData:
    """Load one user JSON file; asserts required 'username' key is present."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "username" in data, f"Missing 'username' field in {path}"
    return UserData(
        username=data["username"],
        posts=data.get("posts", []),
        comments=data.get("comments", []),
    )


def build_text_sample(user: UserData, max_chars: int) -> str:
    """Build LLM input: r/ethz content always included; other subreddits only on keyword match."""
    parts: List[str] = []  # text fragments accumulated in chronological order
    for p in user.posts:
        subreddit = p.get("subreddit", "")
        title = p.get("title", "")
        body = p.get("selftext", "").strip()
        if not _is_relevant(subreddit, title, body):
            continue
        prefix = f"[post r/{subreddit}]" if subreddit else "[post]"
        parts.append(f"{prefix} {title}\n{body}" if body else f"{prefix} {title}")
    for c in user.comments:
        subreddit = c.get("subreddit", "")
        body = c.get("body", "").strip()
        if body and _is_relevant(subreddit, body):
            prefix = f"[comment r/{subreddit}]" if subreddit else "[comment]"
            parts.append(f"{prefix} {body}")
    combined = "\n\n".join(parts)
    return combined[:max_chars]


def load_existing_results(path: Path) -> Dict[str, Classification]:
    """Load previously saved per-user classifications from CSV for --resume mode."""
    if not path.exists():
        return {}
    existing: Dict[str, Classification] = {}  # username → Classification
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing[row["username"]] = Classification(
                username=row["username"],
                voice_type=row["voice_type"],
                subtype=row["subtype"],
                confidence=float(row["confidence"]),
                n_posts=int(row["n_posts"]),
                evidence=row.get(
                    "evidence", ""
                ),  # absent in old CSVs — default to empty
                reasoning=row["reasoning"],
            )
    return existing


def save_classifications(results: List[Classification], path: Path) -> None:
    """Write all per-user classification rows to CSV."""
    fields = [
        "username",
        "voice_type",
        "subtype",
        "confidence",
        "n_posts",
        "evidence",
        "reasoning",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "username": r.username,
                    "voice_type": r.voice_type,
                    "subtype": r.subtype,
                    "confidence": round(r.confidence, 4),
                    "n_posts": r.n_posts,
                    "evidence": r.evidence,
                    "reasoning": r.reasoning,
                }
            )
    print(f"Saved {len(results)} classifications → {path}")


def save_distribution(rows: List[Dict], path: Path) -> None:
    """Write aggregated distribution table to CSV (mirrors table2_voice_distribution.csv)."""
    fields = [
        "voice_type",
        "subtype",
        "n",
        "mean_posts",
        "total_posts",
        "mean_confidence",
        "share_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved distribution ({len(rows)} rows) → {path}")


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def _call_ollama(
    model: str, system: str, user_msg: str, options: Dict, fmt: Optional[Dict] = None
) -> str:
    """Shared Ollama call with retries on transient errors; returns stripped content."""
    kwargs: Dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "options": options,
    }
    if fmt is not None:  # grammar-constrained structured output
        kwargs["format"] = fmt
    if "qwen3" in model:  # thinking=False avoids ~900-token CoT preamble per call
        kwargs["think"] = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return ollama.chat(**kwargs).message.content.strip()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [retry {attempt}/{MAX_RETRIES}] {exc}")
            time.sleep(RETRY_DELAY)


def _parse_extraction(raw: str, debug: bool = False) -> List[str]:
    """Parse stage-1 JSON output into a list of self-identifying statement strings."""
    if debug:
        print(f"\n[DEBUG extraction raw]\n{raw}\n")
    try:
        data = json.loads(raw)
        stmts = data["statements"]
        assert isinstance(stmts, list), f"expected list, got {type(stmts)}"
        return [s for s in stmts if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, KeyError, AssertionError) as exc:
        if debug:
            print(f"[DEBUG] extraction parse failed: {exc}")
        return []


def _extract_statements(
    username: str, text_sample: str, model: str, debug: bool = False
) -> List[str]:
    """Stage 1: extract verbatim self-identifying statements from raw user text."""
    msg = EXTRACTION_TEMPLATE.format(username=username, text_sample=text_sample)
    raw = _call_ollama(model, EXTRACTION_SYSTEM, msg, EXTRACTION_OPTIONS, fmt=EXTRACTION_SCHEMA)
    return _parse_extraction(raw, debug=debug)


def _parse_classify(raw: str, debug: bool = False) -> Tuple[str, str, float, str, str]:
    """Parse stage-2 JSON output into (voice_type, subtype, confidence, evidence, reasoning)."""
    if debug:
        print(f"\n[DEBUG classify raw]\n{raw}\n")
    try:
        fields = json.loads(raw)
    except json.JSONDecodeError as exc:
        if debug:
            print(f"[DEBUG] classify parse failed: {exc}")
        return "unknown", "", 0.0, "", "json parse error"

    voice_type = str(fields.get("voice_type", "unknown")).lower()
    raw_sub = str(fields.get("subtype", "")).lower()
    subtype = _SUBTYPE_ALIASES.get(raw_sub, raw_sub)
    evidence = str(fields.get("evidence", ""))
    reasoning = str(fields.get("reasoning", ""))
    try:
        confidence = float(fields.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0

    if voice_type not in VOICE_TYPES:
        if debug:
            print(f"[DEBUG] invalid voice_type={voice_type!r}, falling back to unknown")
        return "unknown", "", 0.0, evidence, reasoning

    if (voice_type, subtype) not in VALID_COMBOS:
        valid_subtypes = {s for (vt, s) in VALID_COMBOS if vt == voice_type}
        if debug:
            print(f"[DEBUG] subtype={subtype!r} not valid for {voice_type!r}, clearing (valid: {valid_subtypes})")
        subtype = ""
        confidence = min(confidence, 0.5)  # penalise lost subtype

    if voice_type == "external_individual" and confidence < 0.5:  # enforce threshold in code
        if debug:
            print(f"[DEBUG] external_individual conf={confidence:.2f} < 0.5, downgrading to unknown")
        return "unknown", "", 0.0, evidence, f"downgraded: external_individual confidence {confidence:.2f} below threshold"

    if not evidence.strip() and voice_type != "unknown":  # unsupported classification → weaken
        confidence = min(confidence, 0.3)

    return voice_type, subtype, confidence, evidence, reasoning


def classify_user(
    username: str, text_sample: str, model: str, n_posts: int, debug: bool = False
) -> Classification:
    """Two-stage classification: extract self-identifying statements then classify."""
    statements = _extract_statements(username, text_sample, model, debug=debug)
    if not statements:
        return Classification(
            username=username,
            voice_type="unknown",
            subtype="",
            confidence=0.0,
            n_posts=n_posts,
            evidence="",
            reasoning="no self-identifying statements found",
        )
    statements_text = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    msg = CLASSIFICATION_TEMPLATE.format(username=username, statements_text=statements_text)
    raw = _call_ollama(model, CLASSIFICATION_SYSTEM, msg, OLLAMA_OPTIONS, fmt=CLASSIFICATION_SCHEMA)
    voice_type, subtype, confidence, evidence, reasoning = _parse_classify(raw, debug=debug)
    return Classification(
        username=username,
        voice_type=voice_type,
        subtype=subtype,
        confidence=confidence,
        n_posts=n_posts,
        evidence=evidence,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_distribution(results: List[Classification]) -> List[Dict]:
    """Group classifications by (voice_type, subtype) and compute summary stats."""
    groups: Dict[Tuple[str, str], List[Classification]] = defaultdict(list)
    for r in results:
        groups[(r.voice_type, r.subtype)].append(r)

    total = len(results)  # denominator for share_pct
    rows: List[Dict] = []
    for (vtype, subtype), group in sorted(groups.items()):
        n = len(group)
        total_posts = sum(r.n_posts for r in group)
        rows.append(
            {
                "voice_type": vtype,
                "subtype": subtype,
                "n": n,
                "mean_posts": round(total_posts / n, 2) if n else 0,
                "total_posts": total_posts,
                "mean_confidence": round(sum(r.confidence for r in group) / n, 4)
                if n
                else 0,
                "share_pct": round(100 * n / total, 4) if total else 0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify ETH Zurich Reddit users into voice typology categories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="results/user_posts",
        help="Directory of user JSON files (default: results/user_posts)",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--model", default="qwen3:14b", help="Ollama model name (default: qwen3)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max chars of text fed to LLM per user (default: 1200)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip users already present in the output CSV",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw LLM output and parse decisions for each user",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_files = sorted(input_dir.glob("*.json"))
    assert user_files, f"No JSON files found in {input_dir!r}"
    print(f"Found {len(user_files)} user files in {input_dir}/")

    classifications_path = (
        output_dir / "user_voice_classifications.csv"
    )  # per-user output
    distribution_path = output_dir / "user_voice_distribution.csv"  # aggregated output

    existing = load_existing_results(classifications_path) if args.resume else {}
    if existing:
        print(f"Resuming — {len(existing)} users already classified, skipping them")

    results: List[Classification] = list(
        existing.values()
    )  # seed with prior run if resuming
    to_process = [f for f in user_files if f.stem not in existing]

    for i, fpath in enumerate(tqdm(to_process, desc="Classifying users", unit="user"), 1):
        user = load_user_data(fpath)
        text_sample = build_text_sample(user, args.max_chars)
        n_posts = len(user.posts) + len(user.comments)

        if not text_sample.strip():  # user has no text content — classify as unknown
            results.append(
                Classification(
                    username=user.username,
                    voice_type="unknown",
                    subtype="",
                    confidence=0.0,
                    n_posts=n_posts,
                    evidence="",
                    reasoning="no text content available",
                )
            )
        else:
            clf = classify_user(
                user.username, text_sample, args.model, n_posts, debug=args.debug
            )
            results.append(clf)

        if i % 50 == 0:  # checkpoint every 50 users to guard against mid-run crashes
            save_classifications(results, classifications_path)

    save_classifications(results, classifications_path)
    dist_rows = aggregate_distribution(results)
    save_distribution(dist_rows, distribution_path)

    print(
        f"\nDone. {len(results)} users classified into {len(dist_rows)} voice type/subtype groups."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
