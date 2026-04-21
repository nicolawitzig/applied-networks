"""
Step 3: pull the 2021 tweets for every kept account.

Usage:
    python scripts/03_scrape_tweets.py

Appends to data/raw/tweets.jsonl. Safe to re-run: already-scraped accounts
are skipped based on a sidecar progress file.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twscrape import API

from src.scrapers.twscrape_client import fetch_tweets
from src.utils.io import load_config, read_csv, resolve_path


async def main() -> None:
    cfg = load_config()
    accounts_csv = resolve_path(cfg["paths"]["processed"]) / "accounts.csv"
    df = read_csv(accounts_csv)

    since = datetime.fromisoformat(cfg["time_window"]["start"])
    until = datetime.fromisoformat(cfg["time_window"]["end"])
    per_account_limit = cfg["scraping"].get("tweets_per_account_limit")

    out_path = resolve_path(cfg["paths"]["raw"]) / "tweets.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = out_path.with_suffix(".progress.json")
    done: set[str] = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()

    api = API()
    written = 0
    with open(out_path, "a", encoding="utf-8") as sink:
        for _, row in df.iterrows():
            user_id = str(row["user_id"])
            handle = row["handle"]
            if user_id in done:
                continue
            try:
                tweets = await fetch_tweets(
                    api, int(user_id), since=since, until=until, limit=per_account_limit,
                )
            except Exception as exc:
                print(f"[03] @{handle}: failed ({exc})")
                continue
            for t in tweets:
                sink.write(json.dumps(asdict(t), ensure_ascii=False, default=str) + "\n")
                written += 1
            done.add(user_id)
            progress_path.write_text(json.dumps(sorted(done)))
            print(f"[03] @{handle}: {len(tweets)} tweets")

    print(f"[03] total new tweets written: {written}")


if __name__ == "__main__":
    asyncio.run(main())
