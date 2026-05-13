"""
Step 5: topic + tonality for tweets that mention the official accounts, plus
Table 2 / Table 3 summaries.

Usage:
    python scripts/05_content_analysis.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.content import classify_tonality, classify_topic, mentions_any
from src.analysis.stats import interaction_with_official, voice_distribution
from src.utils.io import load_config, read_csv, read_jsonl, resolve_path, write_csv


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["processed"])
    raw = resolve_path(cfg["paths"]["raw"])

    accounts = read_csv(processed / "accounts.csv")
    tweets = pd.DataFrame(read_jsonl(raw / "tweets.jsonl"))

    official_handles = [e["handle"] for e in cfg["official_accounts"]]

    # Table 2 equivalent.
    dist = voice_distribution(accounts)
    write_csv(dist, processed / "table2_voice_distribution.csv")

    # Table 3 equivalent.
    inter = interaction_with_official(tweets, official_handles, accounts)
    write_csv(inter, processed / "table3_interaction.csv")

    # Table 4 equivalent: topics/tonality of tweets that mention official accounts.
    mask = tweets.apply(
        lambda r: mentions_any(r.get("text", "") or "", official_handles)
        or any(h.lower() in {h2.lower() for h2 in official_handles} for h in r.get("mentioned_handles", []) or []),
        axis=1,
    )
    mentions_df = tweets[mask].copy()
    mentions_df["topic"] = mentions_df["text"].fillna("").apply(classify_topic)
    mentions_df["tonality"] = mentions_df["text"].fillna("").apply(classify_tonality)

    merged = mentions_df.merge(
        accounts[["handle", "voice_type", "subtype"]], on="handle", how="left"
    )
    table4 = (
        merged.groupby(["voice_type", "subtype"], dropna=False)
        .agg(
            n=("tweet_id", "count"),
            pct_academic=("topic", lambda s: 100 * (s == "academic").mean()),
            pct_organizational=("topic", lambda s: 100 * (s == "organizational").mean()),
            pct_other=("topic", lambda s: 100 * (s == "other").mean()),
            pct_positive=("tonality", lambda s: 100 * (s == "positive").mean()),
            pct_neutral=("tonality", lambda s: 100 * (s == "neutral").mean()),
            pct_negative=("tonality", lambda s: 100 * (s == "negative").mean()),
        )
        .reset_index()
    )
    write_csv(table4, processed / "table4_topics_tonality.csv")
    print("[05] wrote Table 2, 3, 4 equivalents to data/processed/")


if __name__ == "__main__":
    main()
