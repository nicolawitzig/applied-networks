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
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import ollama
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Typology constants
# ---------------------------------------------------------------------------

VOICE_TYPES = {  # all valid top-level voice type labels
    "central_institutional",
    "decentral_individual",
    "decentral_institutional",
    "former_individual",
    "former_institutional",
    "external_individual",  # affiliated with a university other than ETH
    "unknown",
}

VALID_COMBOS: frozenset = frozenset(
    {  # (voice_type, subtype) pairs from Volk et al.
        ("central_institutional", "administrative_body"),
        ("decentral_individual", "phd"),
        ("decentral_individual", "postdoc"),
        ("decentral_individual", "professor"),
        ("decentral_individual", "researcher"),
        ("decentral_individual", "student"),
        ("decentral_institutional", "department_or_lab"),
        ("former_individual", "employee_or_alumni"),
        ("former_individual", "postdoc"),
        ("former_individual", "phd"),
        ("former_individual", "professor"),
        ("former_individual", "researcher"),
        ("former_individual", "student"),
        ("former_institutional", "unit"),
        ("external_individual", "student"),
        ("external_individual", "phd"),
        ("external_individual", "postdoc"),
        ("external_individual", "professor"),
        ("external_individual", "researcher"),
        ("external_individual", "other"),
        ("unknown", ""),
    }
)

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
# Ollama inference config  (mirrors LLM_categorization.py)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3  # attempts per user before giving up and returning unknown
RETRY_DELAY = 5  # seconds between retries on transient Ollama errors

OLLAMA_OPTIONS = {  # classify call: needs room for CoT think-block + five output lines
    "temperature": 0,
    "num_predict": 2048,
    "num_ctx": 4096,
}

EXTRACTION_OPTIONS = {  # extract call: output is a short bullet list
    "temperature": 0,
    "num_predict": 512,
    "num_ctx": 4096,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You extract self-identification statements from Reddit posts.
Find every sentence where THIS USER describes their own current or former university
affiliation, role, or academic status. Copy each statement verbatim, one per line,
prefixed with "- ".

Rules:
  - Only include statements about THIS USER's own situation (first person).
  - Do NOT include questions, general facts about universities, or statements about others.
  - Do NOT include the username as evidence.
  - If no such statements exist, output exactly: NONE
"""

EXTRACTION_TEMPLATE = """Extract self-identification statements from the posts and comments below.

## Username
{username}

## Posts and comments
{text_sample}

Output each verbatim statement on its own line prefixed with "- ", or output NONE."""

SYSTEM_PROMPT = """You are an expert coder for a communication science study on Reddit accounts.
You will receive pre-extracted self-identification statements from a Reddit user in r/ethz.
Classify the account using the voice typology from "The Plurivocal University" (Volk et al., 2024).

## Step 1 — Decision tree (follow in order, stop at first match)

CHECK A — Does any statement mention a university OTHER than ETH (e.g. TU Munich, EPFL,
ZHAW, UZH, MIT, any other institution) as the user's own affiliation?
  → YES: voice_type = "external_individual", subtype = their role there. STOP.

CHECK B — Does any statement show a current or past connection to ETH Zurich?
  → YES: assign the appropriate ETH voice type from Step 2. STOP.

CHECK C — No university affiliation can be determined from the statements.
  → voice_type = "unknown".

## Step 2 — Voice type (only for ETH-affiliated accounts)

### central_institutional / administrative_body
Official ETH-wide administrative or service units: central admin, registrar, press office,
career centre, international office, student services, library.

### decentral_individual
Individual academics or students currently affiliated with ETH.
Subtypes:
  professor  — full/associate/assistant professor or lecturer at ETH
  postdoc    — postdoctoral researcher at ETH
  phd        — PhD candidate / doctoral student at ETH
  researcher — research associate, scientist, non-faculty researcher at ETH
  student    — undergraduate or master's student at ETH

### decentral_institutional / department_or_lab
Official ETH departments, institutes, labs, research groups, or centres.

### former_individual
Former ETH employees, alumni, or researchers no longer at ETH.
Subtypes: professor | postdoc | phd | researcher | student | employee_or_alumni

### former_institutional / unit
Former ETH units or organisations that used to be part of ETH.

### external_individual
Individual explicitly affiliated with a non-ETH university (EPFL, ZHAW, UZH, etc.).
Subtypes: professor | postdoc | phd | researcher | student | other

### unknown
No university affiliation can be determined from the text.

## Step 3 — Evidence

Pick the single strongest statement and copy it verbatim into "evidence".
If no statement supports affiliation, set evidence to "" and voice_type to "unknown".

## Output format

Respond with ONLY these five lines — no explanation, no markdown, no extra text:
VOICE_TYPE: <voice_type>
SUBTYPE: <subtype or blank>
CONFIDENCE: <float 0.0–1.0>
EVIDENCE: <verbatim quote from the statements, or blank>
REASONING: <one short sentence citing the evidence>

Confidence scale:
  0.8–1.0 — explicit self-identification ("I am a PhD student at ETH")
  0.5–0.7 — strong contextual signal (own courses, exams, lab work)
  0.2–0.4 — weak signal, plausible but uncertain
  0.0     — no evidence; voice_type must be "unknown"
"""

CLASSIFICATION_TEMPLATE = """Classify the following Reddit account using the extracted statements below.

## Account
Username: {username}

## Self-identification statements
{statements}

CHECK A — non-ETH university? → external_individual
CHECK B — ETH affiliation? → appropriate ETH voice type
CHECK C — no affiliation? → unknown
Respond ONLY with the five KEY: value lines."""

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
    """Concatenate post titles + bodies + comment bodies with subreddit, truncated to max_chars."""
    parts: List[str] = []  # text fragments accumulated in chronological order
    for p in user.posts:
        subreddit = p.get("subreddit", "")
        title = p.get("title", "")
        body = p.get("selftext", "").strip()
        prefix = f"[post r/{subreddit}]" if subreddit else "[post]"
        parts.append(f"{prefix} {title}\n{body}" if body else f"{prefix} {title}")
    for c in user.comments:
        subreddit = c.get("subreddit", "")
        body = c.get("body", "").strip()
        if body:
            prefix = f"[comment r/{subreddit}]" if subreddit else "[comment]"
            parts.append(f"{prefix} {body}")
    combined = "\n\n".join(parts)
    return combined[:max_chars]


def build_prompt(username: str, statements: str) -> str:
    """Fill the classification template with pre-extracted affiliation statements."""
    return CLASSIFICATION_TEMPLATE.format(username=username, statements=statements)


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


def _call_ollama(model: str, system: str, user_msg: str, options: Dict) -> str:
    """Shared Ollama call with retries on transient errors; returns stripped content."""
    kwargs: Dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "options": options,
    }
    if "qwen3" in model:  # CoT thinking mode improves ambiguous cases
        kwargs["think"] = True
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return ollama.chat(**kwargs).message.content.strip()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [retry {attempt}/{MAX_RETRIES}] {exc}")
            time.sleep(RETRY_DELAY)


def _parse_extraction(raw: str) -> str:
    """Return bullet lines from extraction output, or 'NONE' if none found."""
    bullets = [l.strip() for l in raw.splitlines() if l.strip().startswith("-")]
    return "\n".join(bullets) if bullets else "NONE"


def _parse_classify(raw: str) -> Tuple[str, str, float, str, str]:
    """Parse KEY: value lines into (voice_type, subtype, confidence, evidence, reasoning)."""
    fields: Dict[str, str] = {}  # uppercase key → value
    for line in raw.splitlines():
        if ": " in line:
            key, _, val = line.partition(": ")
            fields[key.strip().upper()] = val.strip()

    voice_type = fields.get("VOICE_TYPE", "unknown").lower()
    subtype = fields.get("SUBTYPE", "").lower()
    evidence = fields.get("EVIDENCE", "")
    reasoning = fields.get("REASONING", "")
    try:
        confidence = float(fields.get("CONFIDENCE", "0.0"))
    except ValueError:
        confidence = 0.0

    if voice_type not in VOICE_TYPES:
        voice_type, subtype, confidence = "unknown", "", 0.0
    if (voice_type, subtype) not in VALID_COMBOS:
        subtype = ""
        if (voice_type, subtype) not in VALID_COMBOS:
            voice_type, subtype, confidence = "unknown", "", 0.0

    return voice_type, subtype, confidence, evidence, reasoning


def extract_statements(username: str, text_sample: str, model: str) -> str:
    """Pass 1 — extract verbatim self-identification statements; returns bullets or 'NONE'."""
    msg = EXTRACTION_TEMPLATE.format(username=username, text_sample=text_sample)
    raw = _call_ollama(model, EXTRACTION_SYSTEM, msg, EXTRACTION_OPTIONS)
    return _parse_extraction(raw)


def classify_user(username: str, text_sample: str, model: str, n_posts: int) -> Classification:
    """Pass 2 — classify based on extracted statements; returns unknown if none found."""
    statements = extract_statements(username, text_sample, model)
    if statements == "NONE":
        return Classification(username=username, voice_type="unknown", subtype="",
                              confidence=0.0, n_posts=n_posts, evidence="",
                              reasoning="no affiliation statements found")
    raw = _call_ollama(model, SYSTEM_PROMPT, build_prompt(username, statements), OLLAMA_OPTIONS)
    voice_type, subtype, confidence, evidence, reasoning = _parse_classify(raw)
    return Classification(username=username, voice_type=voice_type, subtype=subtype,
                          confidence=confidence, n_posts=n_posts, evidence=evidence,
                          reasoning=reasoning)


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
        "--model", default="llama3.2", help="Ollama model name (default: llama3.2)"
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

    for fpath in tqdm(to_process, desc="Classifying users", unit="user"):
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
            continue

        clf = classify_user(user.username, text_sample, args.model, n_posts)
        results.append(clf)

    save_classifications(results, classifications_path)
    dist_rows = aggregate_distribution(results)
    save_distribution(dist_rows, distribution_path)

    print(
        f"\nDone. {len(results)} users classified into {len(dist_rows)} voice type/subtype groups."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
