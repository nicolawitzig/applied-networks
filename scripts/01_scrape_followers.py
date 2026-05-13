"""
Step 1: list followers of the official university accounts.

Usage:
    python scripts/01_scrape_followers.py

Reads config/config.yaml, writes data/raw/followers_<handle>.jsonl.

Requires twscrape accounts to be pre-configured in its local DB:
    twscrape add_accounts accounts.txt login:password:email:email_password
    twscrape login_accounts
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twscrape import API

from src.scrapers.twscrape_client import list_followers
from src.utils.io import load_config, write_jsonl


async def main() -> None:
    cfg = load_config()
    api = API()
    limit = cfg["scraping"].get("followers_limit")

    for entry in cfg["official_accounts"]:
        handle = entry["handle"]
        print(f"[01] fetching followers of @{handle} (limit={limit})")
        accounts = await list_followers(api, handle, limit=limit)
        out_path = Path(cfg["paths"]["raw"]) / f"followers_{handle}.jsonl"
        write_jsonl([asdict(a) for a in accounts], out_path)
        print(f"[01] wrote {len(accounts)} accounts -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
