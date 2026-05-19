# Reddit Community Analysis Pipeline — ETH Zurich Case Study

## Overview

This document describes a four-stage pipeline for identifying, profiling, and classifying Reddit users affiliated with a target academic community (r/ethz), and preparing that data for social network / community analysis. Each stage produces structured output consumed directly by the next.

---

## Stage 1 — User Discovery & Cross-Subreddit Mapping

**Script:** `scrape_user_subreddits_crossref.py`

**Purpose:** Exhaustively enumerate every user who posted or commented in the target subreddit within a given date range, then map each user's full Reddit activity across all subreddits.

**Method:**
- Paginates the Arctic Shift Reddit archive API (`arctic-shift.photon-reddit.com`) in descending time order, fetching posts and comments in batches of 100 until the full date range is covered.
- Deduplicates usernames, filtering out deleted accounts and bots.
- For each discovered user, re-queries the API to fetch *all* of their posts and comments across *all* subreddits (not just the target), within the same date window.
- Rate-limited to avoid API overload (~1 req/sec).

**Input:**
- Target subreddit name (e.g. `ethz`)
- Date range (`--after`, `--before` as `YYYY-MM-DD`)

**Output:** Single CSV file — one row per user — with columns:
| Column | Description |
|---|---|
| `username` | Reddit username |
| `total_posts` | Post count in date range |
| `total_comments` | Comment count in date range |
| `subreddit_count` | Number of distinct subreddits active in |
| `first_date` / `last_date` | Activity window |
| `subreddits` | Packed `subreddit:count` list, sorted by frequency |

**Key design choice:** The packed `subreddit:count` format keeps the entire cross-subreddit fingerprint in a single CSV field, enabling fast load and downstream network construction without a separate join.

---

## Stage 2 — Per-User Content Retrieval

**Script:** `scrape_user_posts.py`

**Purpose:** Fetch the full text content (post titles, selftext, comment bodies) for each user, scoped to a specific subreddit and date range, and persist it for offline LLM classification.

**Method:**
- Reads usernames from one or more Stage 1 CSVs (deduplicated across files).
- Parses the date range directly from the CSV filename (pattern: `*_YYYY_Mon_Mon.csv`), removing the need for manual date arguments.
- For each user, paginates both the posts and comments endpoints with deduplication via seen-ID sets to prevent cursor-boundary duplicates.
- Merges with any pre-existing output file (safe to resume mid-run — only new IDs are appended).
- Optional subreddit filter: pass `--subreddit ethz` to restrict content to the target community, or `--subreddit all` for the user's full history.

**Input:**
- Stage 1 crossref CSV file(s)
- Optional subreddit filter and output directory

**Output:** One JSON file per user in `results/user_posts/<username>.json`:
```json
{
  "username": "...",
  "posts":    [{ "id", "title", "selftext", "subreddit", "created_utc", "score", ... }],
  "comments": [{ "id", "body", "subreddit", "created_utc", "score", "link_id", ... }]
}
```

**Key design choice:** One file per user means the pipeline is trivially resumable — a file's existence signals completion. Large batches can be interrupted and restarted without re-fetching.

---

## Stage 3 — LLM Voice-Type Classification

**Script:** `classify_content.py`

**Purpose:** Classify each user into the ETH Zurich voice typology (Volk et al., "The Plurivocal University", 2024) using a two-pass local LLM (Ollama).

**Typology (Volk et al. 2024):**
- `decentral_individual` — current ETH students, PhD candidates, postdocs, researchers, professors
- `central_institutional` — ETH-wide admin/service bodies
- `decentral_institutional` — ETH departments, institutes, labs
- `former_individual` / `former_institutional` — alumni and former units
- `external_individual` — affiliated with a non-ETH university
- `unknown` — no affiliation signal found

**Two-pass LLM method:**
1. **Extraction pass** — A focused prompt extracts verbatim first-person self-identification statements from the user's posts/comments (e.g. "I'm a PhD student at ETH"). Outputs a bullet list or `NONE`.
2. **Classification pass** — A structured decision-tree prompt (Check A: non-ETH? → Check B: ETH? → Check C: unknown?) classifies based solely on the extracted statements. Returns five structured fields: `VOICE_TYPE`, `SUBTYPE`, `CONFIDENCE`, `EVIDENCE`, `REASONING`.

**Why two passes?** Separating extraction from classification reduces hallucination: the classifier only sees curated quotes, not the full noisy Reddit text.

**Input:** Per-user JSON files from Stage 2 (`results/user_posts/*.json`)

**Output:**
- `results/user_voice_classifications.csv` — one row per user with voice type, subtype, confidence score, evidence quote, and reasoning.
- `results/user_voice_distribution.csv` — aggregated distribution table (count, mean activity, mean confidence, share %) mirroring Table 2 in Volk et al.

**Key design choice:** `--resume` flag skips already-classified users, making large batches safe to restart. Confidence scores enable downstream filtering by classification certainty.

---

## Stage 4 — Community & Social Network Analysis *(planned)*

**Status:** Not yet implemented.

**Planned inputs:**
- Stage 1 crossref CSVs → bipartite user–subreddit co-activity graph
- Stage 3 voice classifications → node attributes (type, subtype, confidence)

**Expected analyses:**
- Community detection on the user–subreddit bipartite graph (e.g. modularity-based clustering)
- Cross-community bridging: which voice types link the ETH community to external Reddit communities?
- Influence / centrality measures per voice type
- Temporal activity patterns across the user cohort

---

## End-to-End Data Flow

```
r/ethz + date range
        │
        ▼
[Stage 1] scrape_user_subreddits_crossref.py
        │  ethz_users_crossref.csv
        │  (username, subreddit:count fingerprint)
        ▼
[Stage 2] scrape_user_posts.py
        │  results/user_posts/<username>.json  (one per user)
        │  (full post + comment text, resumable)
        ▼
[Stage 3] classify_content.py
        │  user_voice_classifications.csv
        │  user_voice_distribution.csv
        │  (voice type, subtype, confidence, evidence)
        ▼
[Stage 4] community_network_analysis.py  ← to be built
           (graph construction, community detection, bridging analysis)
```

---

## Technical Notes for Slide Generation

- **Data source:** Arctic Shift public Reddit archive API — covers historical posts/comments without Reddit API rate limits.
- **Scale:** Designed for hundreds to low thousands of users; pagination + deduplication handles API cursor edge cases.
- **LLM backend:** Ollama (local inference) — no API cost, no data egress. Default model: `llama3.2`; `qwen3`-family models enable chain-of-thought thinking mode.
- **Reproducibility:** All stages are idempotent and resumable. Date ranges are encoded in filenames to prevent mis-pairing of inputs.
- **Theoretical grounding:** Classification schema is anchored to a peer-reviewed typology (Volk et al. 2024), enabling comparison with prior findings.
