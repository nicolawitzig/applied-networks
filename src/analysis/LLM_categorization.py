"""
LLM-based voice categorization using a locally running qwen3:14b model via Ollama.

Categorizes each follower into the voice typology from "The Plurivocal Society"
(Brantner et al., 2023) using profile information, tweets authored by the
follower, and tweets where the follower is mentioned.

The first classification gate is UZH affiliation:
  non_uzh  — account has no affiliation with the University of Zurich (UZH).
             This covers other universities, funding bodies, external researchers,
             journalists, and general public followers.

For UZH-affiliated accounts, voice types (Table 1 of the paper):
  official_institutional  — the 2 seeded official UZH accounts
  official_individual     — official UZH spokesperson accounts
  central_institutional   — UZH-wide service units / admin bodies
  central_individual      — individual UZH central admin staff / comms officers
  decentral_institutional — UZH departments, institutes, labs, centres
  decentral_individual    — UZH professors, postdocs, PhDs, researchers, students
  former_institutional    — former UZH units / spinoffs
  former_individual       — UZH alumni, former UZH employees / researchers
  unknown                 — UZH connection suspected but role cannot be determined

Subtypes for decentral_individual / former_individual:
  professor | postdoc | phd | researcher | student | employee_or_alumni

Output: data/categorized/accounts_llm.csv  (same schema as processed/accounts.csv, + method + reasoning)
        data/categorized/table2_llm.csv    (voice distribution summary, Table 2)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FILEPATH = Path(__file__).resolve()
SRC_PATH = FILEPATH.parents[2]
DATA_PATH = SRC_PATH / "data"
RAW_PATH = DATA_PATH / "raw"
CATEGORIZED_PATH = DATA_PATH / "categorized"
CATEGORIZED_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_ACCOUNTS = CATEGORIZED_PATH / "accounts_llm.csv"
OUTPUT_TABLE2 = CATEGORIZED_PATH / "table2_llm.csv"
PROGRESS_FILE = CATEGORIZED_PATH / "progress.jsonl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "qwen3:14b"
MAX_OWN_TWEETS = 8       # authored tweets to include per follower
MAX_MENTION_TWEETS = 5   # tweets mentioning the follower to include
RETRY_DELAY = 5          # seconds between retries on Ollama errors
MAX_RETRIES = 3

# Ollama inference options tuned for speed on a local GPU
OLLAMA_OPTIONS = {
    "temperature": 0,
    "num_predict": 200,   # JSON reply is ~80 tokens; hard-cap prevents rambling
    "num_ctx": 2048,      # our prompts fit in 2 k; smaller KV cache = faster prefill
}

VALID_VOICE_TYPES = {
    "non_uzh",                  # no affiliation with UZH at all
    "official_institutional",
    "official_individual",
    "central_institutional",
    "central_individual",
    "decentral_institutional",
    "decentral_individual",
    "former_institutional",
    "former_individual",
    "unknown",                  # UZH connection suspected but role unclear
}

VALID_SUBTYPES = {
    "professor", "postdoc", "phd", "researcher", "student",
    "employee_or_alumni", "department_or_lab", "administrative_body",
    "unit", "spokesperson",
}

# ---------------------------------------------------------------------------
# Heuristic pre-filter  (mirrors coding.py + bio_filter.py from the paper)
#
# Two-tier decision before touching the LLM:
#   1. Official handles → official_institutional immediately.
#   2. No UZH mention in bio/handle → non_uzh, skip LLM entirely.
#   3. Everything else (UZH-affiliated, role unclear) → LLM.
# ---------------------------------------------------------------------------

# Canonical UZH handles that are always official_institutional
OFFICIAL_HANDLES: set[str] = {"uzh_en", "uzh_ch"}

# Bio needles that confirm UZH affiliation (from config/config.yaml)
UZH_NEEDLES: list[str] = [
    "uzh", "university of zurich", "universität zürich", "uzh.ch",
]
# Regex that catches @UZH_* mentions and bare "uzh" in handles
_UZH_HANDLE_RE = re.compile(r"\buzh\b", re.IGNORECASE)
_UZH_MENTION_RE = re.compile(r"@uzh[_a-z0-9]*", re.IGNORECASE)


def _is_uzh_affiliated(description: str, handle: str) -> bool:
    """True when bio or handle unambiguously points to UZH."""
    handle_lower = handle.lower()
    if any(n in handle_lower for n in UZH_NEEDLES):
        return True
    if _UZH_HANDLE_RE.search(handle):
        return True
    if _UZH_MENTION_RE.search(handle):
        return True
    if not description:
        return False
    desc_lower = description.lower()
    if any(n in desc_lower for n in UZH_NEEDLES):
        return True
    return bool(_UZH_MENTION_RE.search(description))


def heuristic_classify(follower: dict) -> dict | None:
    """
    Returns a label dict when the heuristic is confident, else None (→ LLM).

    Label dict keys: voice_type, subtype, is_institutional, reasoning, method.
    method is 'heuristic' here; the LLM path sets it to 'llm'.
    """
    desc = follower.get("description") or ""
    handle = str(follower.get("handle") or "")

    # Tier 1: seeded official accounts
    if handle.lower() in OFFICIAL_HANDLES:
        return {"voice_type": "official_institutional", "subtype": None,
                "is_institutional": True,
                "reasoning": "heuristic: seeded official UZH account", "method": "heuristic"}

    # Tier 2: UZH affiliation gate — non-UZH accounts skip the LLM entirely
    if not _is_uzh_affiliated(desc, handle):
        return {"voice_type": "non_uzh", "subtype": None,
                "is_institutional": False,
                "reasoning": "heuristic: no UZH mention in bio or handle", "method": "heuristic"}

    # UZH-affiliated but role is unclear — fall through to LLM
    return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert coder for a communication science study on university X accounts.
Your task is to classify a X account using the typology from "The Plurivocal Society"
(Brantner et al., 2023), which studies who speaks on behalf of (or around) the University of Zurich
(UZH / Universität Zürich) on X.

## Step 1 — UZH affiliation check (REQUIRED FIRST)

Ask yourself: does this account have a clear and direct connection to the University of Zurich (UZH)?

Signals of UZH affiliation:
  - Bio or display name explicitly mentions UZH, Universität Zürich, University of Zurich,
    @UZH_en, @UZH_ch, @uzh, or a named UZH unit/department/institute.
  - Handle contains "uzh".
  - Tweets show the person working, studying, or researching at UZH.
  - Account is a UZH organizational unit.

NOT UZH affiliation (use non_uzh):
  - Affiliated with a different university (ETH Zurich, University of Basel, EPFL, Unibern, etc.)
  - Affiliated with a Swiss federal agency, funding body, or think-tank (SNSF, SNF, etc.)
    that is merely a UZH partner or funder but not part of UZH.
  - A general public follower, journalist, company, or NGO with no stated UZH connection.
  - Affiliated with both UZH and another institution but clearly primarily the other institution.

If the account is NOT UZH-affiliated, set voice_type = "non_uzh" and stop — no subtype needed.

## Step 2 — Voice type (only for UZH-affiliated accounts)

### official_institutional
The two official, centrally managed UZH accounts (@UZH_en, @UZH_ch).
Only assign this to those primary institutional accounts.

### official_individual
An officially designated individual spokesperson for UZH as a whole
(rector, president, vice-chancellor acting in an official institutional capacity).

### central_institutional
UZH-wide service or administrative units: libraries, equal-opportunity offices,
alumni services, international office, career center, UZH Graduate Campus, press/media office.

### central_individual
Individual staff working in UZH-wide central administration or communications
(not faculty research staff).

### decentral_institutional
Decentralized UZH academic units: faculties, departments, institutes, research groups, labs,
centers, clinics, or other sub-UZH organizational entities.
Subtype: department_or_lab

### decentral_individual
Individual academics or students currently affiliated with a UZH unit.
Subtypes:
  professor  — full/associate/assistant professor, chair holder at UZH
  postdoc    — postdoctoral researcher at UZH
  phd        — PhD candidate / doctoral student at UZH
  researcher — research associate, scientist, or similar non-faculty researcher at UZH
  student    — undergraduate or master's student at UZH

### former_institutional
Former UZH units, spinoffs, or organizations that used to be part of UZH.
Subtype: unit

### former_individual
Former UZH employees, alumni, or researchers no longer at UZH.
Subtypes: professor | postdoc | phd | researcher | student | employee_or_alumni

### unknown
Account has a plausible UZH connection but the role or current affiliation cannot be determined.

## Output format

Respond with ONLY a JSON object — no explanation, no markdown, no thinking text outside the JSON:
{
  "voice_type": "<one of the 10 voice types above>",
  "subtype": "<subtype or null>",
  "is_institutional": <true or false>,
  "reasoning": "<one short sentence explaining your decision>"
}
"""

CLASSIFICATION_TEMPLATE = """Classify the following X account.

## Account profile
Handle: @{handle}
Display name: {display_name}
Bio: {description}
Followers: {followers_count} | Following: {following_count} | Tweets: {tweet_count}
Account created: {created_at}
Verified: {verified}

## Tweets authored by this account (up to {n_own})
{own_tweets}

## Tweets in which this account is mentioned (up to {n_mentions})
{mention_tweets}

First decide: is this account affiliated with the University of Zurich (UZH)?
If NOT → voice_type must be "non_uzh".
If YES → assign the appropriate UZH voice type from the typology.
Remember: respond ONLY with the JSON object."""


def build_prompt(follower: dict, own_tweets: list[str], mention_tweets: list[str]) -> str:
    own_block = "\n".join(f"  - {t}" for t in own_tweets) if own_tweets else "  (no tweets found)"
    mention_block = "\n".join(f"  - {t}" for t in mention_tweets) if mention_tweets else "  (no mentions found)"
    return CLASSIFICATION_TEMPLATE.format(
        handle=follower.get("handle", ""),
        display_name=follower.get("display_name", ""),
        description=follower.get("description", "") or "",
        followers_count=follower.get("followers_count", 0),
        following_count=follower.get("following_count", 0),
        tweet_count=follower.get("tweet_count", 0),
        created_at=follower.get("created_at", ""),
        verified=follower.get("verified", False),
        n_own=len(own_tweets),
        own_tweets=own_block,
        n_mentions=len(mention_tweets),
        mention_tweets=mention_block,
    )


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------
def classify_with_llm(prompt: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                think=False,
                options=OLLAMA_OPTIONS,
            )
            raw = response["message"]["content"].strip()
            # Strip markdown code fences if the model wraps the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            voice_type = result.get("voice_type", "unknown")
            subtype = result.get("subtype")
            if voice_type not in VALID_VOICE_TYPES:
                voice_type = "unknown"
            if subtype and subtype not in VALID_SUBTYPES:
                subtype = None
            return {
                "voice_type": voice_type,
                "subtype": subtype,
                "is_institutional": bool(result.get("is_institutional", False)),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError) as exc:
            if attempt == MAX_RETRIES:
                print(f"    [warn] JSON parse failed after {MAX_RETRIES} attempts: {exc}")
                return {"voice_type": "unknown", "subtype": None, "is_institutional": False, "reasoning": "parse error"}
            time.sleep(RETRY_DELAY)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"    [warn] Ollama error after {MAX_RETRIES} attempts: {exc}")
                return {"voice_type": "unknown", "subtype": None, "is_institutional": False, "reasoning": str(exc)}
            print(f"    [retry {attempt}] {exc}")
            time.sleep(RETRY_DELAY)


# ---------------------------------------------------------------------------
# Summary table (mirrors Table 2)
# ---------------------------------------------------------------------------
def build_summary(accounts: pd.DataFrame) -> pd.DataFrame:
    grp = accounts.groupby(["voice_type", "subtype"], dropna=False)
    out = grp.agg(
        n=("handle", "count"),
        mean_followers=("followers_count", "mean"),
        total_followers=("followers_count", "sum"),
        mean_tweets=("tweet_count", "mean"),
        total_tweets=("tweet_count", "sum"),
    ).reset_index()
    out["share_pct"] = 100 * out["n"] / out["n"].sum()
    return out.sort_values(["voice_type", "n"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading followers from {RAW_PATH / 'followers_UZH_en.jsonl'} ...")
    followers_df = pd.read_json(RAW_PATH / "followers_UZH_en.jsonl", lines=True)
    print(f"  {len(followers_df)} followers loaded.")

    print(f"Loading tweets from {RAW_PATH / 'tweets.jsonl'} ...")
    tweets_df = pd.read_json(RAW_PATH / "tweets.jsonl", lines=True)
    print(f"  {len(tweets_df)} tweets loaded.")

    # Index tweets by author handle (lowercase)
    tweets_df["handle_lower"] = tweets_df["handle"].str.lower()

    # Build a lookup: handle -> list of tweet texts (own tweets)
    own_tweets_map: dict[str, list[str]] = (
        tweets_df.groupby("handle_lower")["text"]
        .apply(list)
        .to_dict()
    )

    # Build a lookup: handle -> list of tweet texts where they are mentioned
    # mentioned_handles is a list column; we explode it
    if "mentioned_handles" in tweets_df.columns:
        mentions_exploded = tweets_df.explode("mentioned_handles").copy()
        mentions_exploded = mentions_exploded.dropna(subset=["mentioned_handles"])
        mentions_exploded["mentioned_lower"] = mentions_exploded["mentioned_handles"].str.lower()
        mention_tweets_map: dict[str, list[str]] = (
            mentions_exploded.groupby("mentioned_lower")["text"]
            .apply(list)
            .to_dict()
        )
    else:
        mention_tweets_map = {}

    # Load already-completed handles (checkpoint).
    # Delete data/categorized/progress.jsonl to start fresh.
    completed: dict[str, dict] = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        completed[rec["handle"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Resuming — {len(completed)} accounts already classified.")

    results: list[dict] = []
    total = len(followers_df)
    n_heuristic = 0
    n_llm = 0

    progress_fh = open(PROGRESS_FILE, "a", encoding="utf-8")

    try:
        for idx, row in followers_df.iterrows():
            follower = row.to_dict()
            handle = str(follower.get("handle", ""))
            handle_lower = handle.lower()

            # Skip already done
            if handle in completed:
                results.append(completed[handle])
                continue

            print(f"[{idx + 1}/{total}] @{handle} ...", end=" ", flush=True)

            # --- Heuristic pre-filter (tiers 1-3) ---
            label = heuristic_classify(follower)

            if label is not None:
                n_heuristic += 1
                print(f"[heuristic] {label['voice_type']} / {label['subtype']}")
            else:
                # Ambiguous UZH account — ask the LLM
                own = own_tweets_map.get(handle_lower, [])[:MAX_OWN_TWEETS]
                mentions = mention_tweets_map.get(handle_lower, [])[:MAX_MENTION_TWEETS]
                prompt = build_prompt(follower, own, mentions)
                label = classify_with_llm(prompt)
                label["method"] = "llm"
                n_llm += 1
                print(f"[llm]       {label['voice_type']} / {label['subtype']}")

            created_at = follower.get("created_at")
            if hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()

            record = {
                "user_id": follower.get("user_id"),
                "handle": handle,
                "display_name": follower.get("display_name"),
                "description": follower.get("description"),
                "followers_count": follower.get("followers_count"),
                "following_count": follower.get("following_count"),
                "tweet_count": follower.get("tweet_count"),
                "created_at": created_at,
                "verified": follower.get("verified"),
                "voice_type": label["voice_type"],
                "subtype": label["subtype"],
                "is_institutional": label["is_institutional"],
                "method": label["method"],
                "reasoning": label["reasoning"],
            }

            results.append(record)
            progress_fh.write(json.dumps(record) + "\n")
            progress_fh.flush()
    finally:
        progress_fh.close()

    print(f"\nClassification summary: {n_heuristic} heuristic, {n_llm} LLM "
          f"({100 * n_heuristic / max(n_heuristic + n_llm, 1):.1f}% skipped LLM)")

    # Save full accounts CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_ACCOUNTS, index=False)
    print(f"\nSaved {len(out_df)} accounts to {OUTPUT_ACCOUNTS}")

    # Save summary table
    summary = build_summary(out_df)
    summary.to_csv(OUTPUT_TABLE2, index=False)
    print(f"Saved voice distribution summary to {OUTPUT_TABLE2}")
    print("\nVoice distribution:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
