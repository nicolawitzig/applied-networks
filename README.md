# applied-networks

Reproduction of Volk, Vogler, Fürst & Schäfer (2025), *The plurivocal university: Typologizing the diverse voices of a research university on social media*, **Public Understanding of Science** 34(3): 270–290. [DOI](https://doi.org/10.1177/09636625241268700)

## What this is

The paper analyses 619 Twitter accounts tied to the University of Zurich (2021). It bio-filters followers of UZH's two official accounts, hand-codes each into one of 8 "voice" types (Table 1), then runs content + social-network analysis on their 2021 tweets. We try to re-run that pipeline end-to-end.

## What is and isn't reproducible today

- **Scraping.** The paper used the Twitter v1.1 API pre-Musk. That tier is gone; v2 is paywalled. We use [`twscrape`](https://github.com/vladkens/twscrape) as a drop-in that scrapes the web UI via logged-in X accounts. You provide the accounts.
- **Time window.** The paper covers 2021. Twitter's timeline endpoint returns newest-first and the public API caps per-user history at ~3,200 tweets, so historical 2021 coverage may be partial unless you already have archives.
- **Manual coding.** Voice-type, topic, and tonality codes in the paper were done by trained coders. We ship rule-based baselines (`src/scrapers/coding.py`, `src/analysis/content.py`) so the pipeline runs end-to-end; replace with your manual codes in `data/coded/` when you have them.

## Layout

```
├── environment.yml           # conda env
├── config/config.yaml        # target accounts, time window, paths
├── src/
│   ├── scrapers/
│   │   ├── base.py           # AccountRecord / TweetRecord dataclasses
│   │   ├── twscrape_client.py  # thin async wrapper
│   │   ├── bio_filter.py     # keep accounts whose bio mentions UZH
│   │   └── coding.py         # heuristic voice-type pre-coder
│   ├── analysis/
│   │   ├── stats.py          # Table 2 / Table 3 equivalents
│   │   ├── network.py        # mention graph + Louvain
│   │   └── content.py        # topic + tonality baselines
│   └── utils/io.py           # config + JSONL/CSV helpers
├── scripts/                  # 01..05 pipeline entry points
├── data/{raw,processed,coded}/  # empty, .gitignored
└── notebooks/                # for exploration
```

## Setup

```bash
conda env create -f environment.yml
conda activate applied-networks

# one-time: configure twscrape's account pool
twscrape add_accounts accounts.txt username:password:email:email_password
twscrape login_accounts
```

`accounts.txt` is a list of X accounts you own (throwaway test accounts are fine but must be warmed up before scraping at volume). See the [twscrape docs](https://github.com/vladkens/twscrape) for the exact format.

## Pipeline

```bash
python scripts/01_scrape_followers.py    # data/raw/followers_<handle>.jsonl
python scripts/02_filter_accounts.py     # data/processed/accounts.csv
python scripts/03_scrape_tweets.py       # data/raw/tweets.jsonl (resumable)
python scripts/04_build_network.py       # mention_network.graphml + community_summary.csv
python scripts/05_content_analysis.py    # table2/3/4 CSVs
```

`config/config.yaml` controls the official handles, time window, and scrape limits. Start with `followers_limit: 500` for a smoke test before committing to a full run.

## Known gaps to fill

- **Official handles in `config.yaml` need verification.** UZH's current English account is `@UZH_en`; the paper used an English + a "national language" (German) account — confirm the German handle before a real run.
- **Coding baselines are weak.** Replace the regex classifier in `coding.py` and the lexicons in `content.py` with manual codes or a supervised model trained on the paper's codebook (SM2).
- **Viz.** No ForceAtlas2 / Gephi-style visualisation yet; the GraphML output opens in Gephi directly.
- **Intercoder reliability** (Krippendorff's α) not computed — add once you have ≥2 coded samples.
