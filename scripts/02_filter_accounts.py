"""
Step 2: de-duplicate the union of followers, filter by bio mention + activity,
and pre-code voice type.

Usage:
    python scripts/02_filter_accounts.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scrapers.base import AccountRecord
from src.scrapers.bio_filter import filter_accounts
from src.scrapers.coding import classify
from src.utils.io import load_config, read_jsonl, resolve_path, write_csv


def main() -> None:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw"])
    needles = cfg["university"]["short_names"]

    all_rows: dict[str, dict] = {}
    for entry in cfg["official_accounts"]:
        path = raw_dir / f"followers_{entry['handle']}.jsonl"
        if not path.exists():
            print(f"[02] missing {path}, skip")
            continue
        for row in read_jsonl(path):
            all_rows[row["user_id"]] = row

    accounts = [AccountRecord(**r) for r in all_rows.values()]
    print(f"[02] union of followers: {len(accounts)}")

    kept = filter_accounts(accounts, needles=needles, require_active=True, min_tweets=1)
    print(f"[02] kept after bio+activity filter: {len(kept)}")

    rows = []
    for a in kept:
        label = classify(a.description, a.handle, a.display_name)
        rows.append({
            **a.__dict__,
            "voice_type": label["voice_type"],
            "subtype": label["subtype"],
            "is_institutional": label["is_institutional"],
        })

    # Tag the two official accounts explicitly.
    official = {e["handle"].lower() for e in cfg["official_accounts"]}
    df = pd.DataFrame(rows)
    mask = df["handle"].str.lower().isin(official)
    df.loc[mask, "voice_type"] = "official_institutional"
    df.loc[mask, "subtype"] = "official_account"
    df.loc[mask, "is_institutional"] = True

    out = resolve_path(cfg["paths"]["processed"]) / "accounts.csv"
    write_csv(df, out)
    print(f"[02] wrote {out} ({len(df)} rows)")
    print(df["voice_type"].value_counts())


if __name__ == "__main__":
    main()
