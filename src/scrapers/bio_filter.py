"""
Paper's sampling rule: keep a follower account only if the university is
mentioned/tagged in its bio (self-description). Also exclude inactive and
locked accounts.
"""
from __future__ import annotations

import re
from .base import AccountRecord


def mentions_university(description: str, needles: list[str]) -> bool:
    if not description:
        return False
    text = description.lower()
    for n in needles:
        n_low = n.lower()
        if n_low in text:
            return True
    # also catch @UZH_* style handles embedded in bios
    handle_pattern = re.compile(r"@uzh[_a-z0-9]*", re.IGNORECASE)
    if handle_pattern.search(description):
        return True
    return False


def filter_accounts(
    accounts: list[AccountRecord],
    needles: list[str],
    require_active: bool = True,
    min_tweets: int = 1,
) -> list[AccountRecord]:
    kept: list[AccountRecord] = []
    for a in accounts:
        if not mentions_university(a.description, needles):
            continue
        if require_active and a.tweet_count < min_tweets:
            continue
        kept.append(a)
    return kept
