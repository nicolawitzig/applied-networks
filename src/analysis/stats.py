"""
Descriptive stats for RQ1 (reach + activity by voice type) — mirrors Table 2.
"""
from __future__ import annotations

import pandas as pd


def voice_distribution(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    accounts: DataFrame with at least ['voice_type', 'subtype',
              'followers_count', 'tweet_count'].
    Returns one row per (voice_type, subtype) with counts and means,
    analogous to Table 2 of the paper.
    """
    grp = accounts.groupby(["voice_type", "subtype"], dropna=False)
    out = grp.agg(
        n=("handle", "count"),
        mean_followers=("followers_count", "mean"),
        total_followers=("followers_count", "sum"),
        mean_tweets=("tweet_count", "mean"),
        total_tweets=("tweet_count", "sum"),
    ).reset_index()
    out["share_pct"] = 100 * out["n"] / out["n"].sum()
    return out.sort_values(["voice_type", "n"], ascending=[True, False])


def interaction_with_official(
    tweets: pd.DataFrame,
    official_handles: list[str],
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replica of Table 3: mentions / retweets of the official accounts by voice.

    tweets: DataFrame with ['handle', 'mentioned_handles', 'is_retweet',
            'retweeted_handle' (optional), 'in_reply_to_handle' (optional)].
    """
    official_lower = {h.lower() for h in official_handles}

    def _mentions_official(handles: list[str]) -> bool:
        return any((h or "").lower() in official_lower for h in handles or [])

    t = tweets.copy()
    t["mentions_official"] = t["mentioned_handles"].apply(_mentions_official)
    t["retweets_official"] = t.get("retweeted_handle", "").fillna("").str.lower().isin(official_lower)

    per_account = (
        t.groupby("handle")
        .agg(
            mentions_official_count=("mentions_official", "sum"),
            retweets_official_count=("retweets_official", "sum"),
        )
        .reset_index()
    )
    merged = accounts.merge(per_account, on="handle", how="left").fillna(
        {"mentions_official_count": 0, "retweets_official_count": 0}
    )
    summary = merged.groupby(["voice_type", "subtype"], dropna=False).agg(
        n=("handle", "count"),
        mentions=("mentions_official_count", "sum"),
        retweets=("retweets_official_count", "sum"),
    ).reset_index()
    total_mentions = summary["mentions"].sum() or 1
    total_retweets = summary["retweets"].sum() or 1
    summary["mentions_pct"] = 100 * summary["mentions"] / total_mentions
    summary["retweets_pct"] = 100 * summary["retweets"] / total_retweets
    return summary
