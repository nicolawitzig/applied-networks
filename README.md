# X/Twitter Voice Typology Pipeline

Reproduces and extends the Volk et al. (2024) university voice typology methodology ("The Plurivocal Society") using X/Twitter data scraped via [twscrape](https://github.com/vladkens/twscrape). The pipeline identifies who speaks about the University of Zurich (UZH) on X/Twitter, classifies their institutional role, and maps how they interact. An optional LLM-based categorisation path replaces rule-based heuristics with a locally running language model for improved accuracy on ambiguous accounts.

**Prerequisites:** X/Twitter accounts must be pre-configured in twscrape's local SQLite database before running any scraping steps:

```bash
twscrape add_accounts accounts.txt login:password:email:email_password
twscrape login_accounts
```

---

## Step 1 — Follower Discovery

**Script:** `scripts/01_scrape_followers.py`  
**Output:** `data/raw/followers_<handle>.jsonl`

Lists all followers of the official UZH accounts specified in `config/config.yaml` (e.g. `@UZH_en`, `@UZH_ch`). Each account record contains profile metadata: handle, display name, bio description, follower/following counts, tweet count, and creation date. The follower limit is controlled by `scraping.followers_limit` in the config.

```bash
python scripts/01_scrape_followers.py
```

---

## Step 2 — Account Filtering & Heuristic Classification

**Script:** `scripts/02_filter_accounts.py`  
**Output:** `data/processed/accounts.csv`

Merges and de-duplicates follower lists from all official sources, then filters to accounts whose bio mentions UZH (or a configured keyword variant) and have at least one tweet. Each surviving account is assigned a preliminary voice type using a rule-based classifier (`src/scrapers/coding.py`) that matches keywords and regex patterns against the account's bio, handle, and display name. Official seeded accounts are tagged `official_institutional` directly.

Voice types assigned: `decentral_individual` (with subtypes professor / postdoc / phd / researcher / student), `decentral_institutional`, `central_institutional`, `central_individual`, `former_individual`, `former_institutional`, `official_institutional`, `unknown`.

```bash
python scripts/02_filter_accounts.py
```

---

## Step 3 — Tweet Scraping

**Script:** `scripts/03_scrape_tweets.py`  
**Output:** `data/raw/tweets.jsonl`, `data/raw/tweets.jsonl.progress.json`

Fetches all tweets and retweets from the accounts kept after step 2, within the date window set in `config/config.yaml` (e.g. the full year 2021). Each tweet record captures text, language, engagement metrics (likes, retweets, replies, quotes), reply threading (`in_reply_to_user_id`), and extracted `@mention` handles. The script runs asynchronously and checkpoints progress to a `.progress.json` sidecar file so interrupted runs can be safely resumed without re-fetching completed accounts.

```bash
python scripts/03_scrape_tweets.py
```

---

## Step 4a — Mention Network (Heuristic Accounts)

**Script:** `scripts/04_build_network.py`  
**Output:** `data/processed/mention_network.graphml`, `data/processed/community_summary.csv`

Builds a directed weighted graph from the scraped tweets using the heuristic-classified accounts from step 2. Nodes are accounts; directed edges are drawn whenever one account `@mentions` another, weighted by mention count. Louvain community detection runs on the undirected projection of the graph to identify clusters of interacting accounts. The community summary CSV reports the size, dominant voice types, and mean follower count for each detected community.

```bash
python scripts/04_build_network.py
```

---

## Step 4b — Mention Network (LLM Accounts)

**Script:** `scripts/04_LLM_build_network.py`  
**Output:** `data/categorized/mention_network_llm2.graphml`, `data/categorized/community_summary_llm2.csv`

Identical network construction to step 4a, but uses the LLM-categorised account table (`data/categorized/accounts_llm.csv`) produced by the LLM categorisation module (see below). Accounts classified as `non_uzh` are excluded before graph construction, leaving only UZH-affiliated voices. This variant typically yields a larger and more accurately labelled node set than the heuristic path.

```bash
python scripts/04_LLM_build_network.py
```

---

## Step 5 — Content Analysis

**Script:** `scripts/05_content_analysis.py`  
**Output:** `data/processed/table2_voice_distribution.csv`, `table3_interaction.csv`, `table4_topics_tonality.csv`

Produces the three summary tables that mirror the paper's empirical results:

- **Table 2** — Voice type distribution: account counts, mean/total follower and tweet counts per voice type and subtype.
- **Table 3** — Interaction with official accounts: how many accounts of each voice type mention or retweet the official UZH handles, and at what rate.
- **Table 4** — Topic and tonality of tweets directed at official accounts: rule-based lexicon classifiers tag each tweet as `academic` / `organizational` / `other` and `positive` / `neutral` / `negative`.

```bash
python scripts/05_content_analysis.py
```

---

## LLM Categorisation Module

**Script:** `src/analysis/LLM_categorization.py`  
**Output:** `data/categorized/accounts_llm.csv`, `data/categorized/table2_llm.csv`

Replaces the heuristic classifier from step 2 with a locally running `qwen3:14b` model served through [Ollama](https://ollama.com) for better handling of ambiguous accounts. Classification proceeds in two tiers to minimise unnecessary LLM calls:

1. **Fast-path heuristics** — Official seeded accounts are tagged `official_institutional` immediately. Accounts with no UZH signal in their bio or handle are tagged `non_uzh` and skipped. These two rules bypass the LLM for roughly 96–97 % of all followers.
2. **LLM classification** — The remaining UZH-affiliated accounts are passed to the model. The prompt provides up to 8 of the account's own tweets and up to 5 tweets that mention them, together with the full voice typology definition from Brantner et al. (2023). The model returns a structured JSON object containing `voice_type`, `subtype`, `is_institutional`, and a free-text `reasoning` field. Temperature is fixed at 0 for deterministic output.

The output CSV extends the standard `accounts.csv` schema with `method` (heuristic vs. llm) and `reasoning` columns for full auditability. A `progress.jsonl` checkpoint file allows interrupted runs to resume without reprocessing completed accounts.

An Ollama server with `qwen3:14b` must be running before executing this module:

```bash
ollama serve          # in a separate terminal, if not already running
ollama pull qwen3:14b # first run only

python src/analysis/LLM_categorization.py
```

---

## Dependencies

```
twscrape, pandas, networkx, python-louvain, ollama, pyyaml, tqdm
```

Install with:
```bash
conda env create -f environment.yml
conda activate applied-networks
```

---

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


