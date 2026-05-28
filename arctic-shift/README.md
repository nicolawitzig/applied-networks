# Reddit Voice Typology Pipeline

Reproduces and extends the Volk et al. (2024) university voice typology methodology using Reddit data from the [Arctic Shift](https://arctic-shift.photon-reddit.com) archive API. The pipeline identifies who speaks about ETH Zurich on Reddit, classifies their institutional role, and maps how they interact.

---

## Step 1 — User Discovery & Cross-Subreddit Scraping

**Script:** `scrape_user_subreddits_crossref.py`  
**Output:** `subreddit_scrapes/<subreddit>_users_crossref.csv`

Exhaustively paginates through all posts and comments in a target subreddit (e.g. `r/ethz`) within a given date range. For every unique author found, it then queries the API for all of that user's activity across Reddit — not just in the target subreddit — and records which subreddits they posted or commented in and how many times. The result is one CSV row per user with a packed `subreddit:count` field summarising their Reddit footprint.

For example in our case we wanted to get the data for the year 2024 for the subreddit r/ethz :

```bash
python scrape_user_subreddits_crossref.py --subreddit ethz --after 2024-01-01 --before 2025-01-01
```

---

## Step 2 — Post & Comment Text Scraping

**Script:** `scrape_user_posts.py`  
**Output:** `results/user_posts/<username>.json`

Reads the crossref CSVs from step 1 and fetches all the comments and posts that the user made during the specified year. Each user gets a single JSON file containing their posts and comments. Runs with up to 100 concurrent threads and is safe to resume — existing files are merged rather than overwritten, so you can add new date ranges without re-fetching. One has to be mindful of API limits and might want to limit the number of threads to something less.

```bash
python scrape_user_posts.py --input-dir subreddit_scrapes --year 2024
```

---

## Step 3 — Voice Type Classification

**Script:** `classify_content.py`  
**Output:** `results/final/user_voice_classifications.csv`, `user_voice_distribution.csv`

Classifies each user into the Volk et al. (2024) voice typology using a local LLM via [Ollama](https://ollama.com). Uses a two-stage pipeline:

1. **Extraction** — a constrained-output prompt extracts verbatim first-person affiliation statements from the user's text (e.g. *"I'm doing my PhD here"*, *"I study at EPFL"*). Non-qualifying content (advice to others, hypotheticals, third-party references) is filtered out.
2. **Classification** — the extracted statements are passed to a second prompt that applies a decision tree to assign a `voice_type` and `subtype`. ETH Zurich is the home institution; current or upcoming affiliation supersedes past ones.

Voice types follow the Volk et al. typology: `decentral_individual` (student, phd, postdoc, researcher, professor, applicant), `central_institutional`, `decentral_institutional`, `former_individual`, `former_institutional`, `external_individual`, and `unknown`. Confidence scores penalise weak signals and missing evidence. Supports `--resume` to checkpoint mid-run.

```bash
python classify_content.py --model qwen3:14b --resume
```

---

## Step 4a — User Interaction Graph

**Script:** `build_interaction_graph.py`  
**Output:** `results/final/interaction_graph.html`, `interaction_graph_nodes.csv`, `voice_interaction_correlation.csv`

Builds a directed weighted graph where nodes are classified users and edges represent interactions. Three interaction types are extracted from the scraped posts:

- **Direct replies** — a comment whose `parent_id` resolves to a post or comment authored by another user.
- **Mentions** — `u/username` references in comment bodies.
- **Co-participation** — two users who both commented on the same post (weighted at 0.5, undirected signal).

Community detection runs Louvain on the undirected projection of the graph. The output HTML visualisation (via [Pyvis](https://pyvis.readthedocs.io)) colours nodes by voice type, sizes them by in-degree, and draws community hulls. A `voice_interaction_correlation.csv` summarises intra- vs. inter-voice interaction fractions.

```bash
python build_interaction_graph.py \
  --classifications results/final/user_voice_classifications.csv \
  --posts-dir results/user_posts \
  --output-dir results/final
```

---

## Step 4b — Subreddit Co-participation Graph

**Script:** `build_subreddit_graph.py`  
**Output:** `results/final/subreddit_graph.html`, `subreddit_graph_nodes.csv`

Builds an undirected weighted graph where nodes are subreddits and edge weights equal the number of classified users active in both subreddits. Only non-unknown users contribute. Nodes are sized by user count and coloured by the dominant voice type of their user base. Louvain community detection groups subreddits by shared audience.

```bash
python build_subreddit_graph.py \
  --classifications results/final/user_voice_classifications.csv \
  --posts-dir results/user_posts \
  --output-dir results/final
```

---

## Dependencies

```
requests, pandas, networkx, python-louvain, pyvis, ollama, tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

An Ollama server with a supported model (e.g. `qwen3:14b`) must be running locally for step 3.
