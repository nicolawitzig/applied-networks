"""
Thin wrapper around twscrape for the three scraping jobs the paper needs:

  1. List followers of the official UZH accounts.
  2. Fetch each follower's profile (for bio-based filtering + coding).
  3. Pull each kept account's tweets in the 2021 window.

twscrape needs one or more X accounts in its local SQLite pool. See README
for the account setup step. This module is only the thinnest possible
async wrapper — batch orchestration lives in scripts/.
"""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

import re

from twscrape import API, gather
import twscrape.xclid as _xclid

from .base import AccountRecord, TweetRecord


# Patch: twscrape 0.17's get_scripts_list splits x.com's JS on 'e=>e+"."+', which
# no longer exists — current x.com emits '[e]||e)+"."+'. Without this patch,
# XClIdGen.create() raises IndexError, which twscrape mis-diagnoses as a rate
# limit and locks the account for 15 minutes. We parse the two relevant JS
# object literals (script-id -> name, script-id -> hash) with a regex instead.
def _patched_get_scripts_list(text: str):
    hash_end = text.find('[e]+"a.js"')
    name_end = text.find('[e]||e)+"."+')
    if hash_end < 0 or name_end < 0:
        raise RuntimeError("x.com JS format changed again — update the parser")

    def _brace_start(end: int, stop: int = -1) -> int:
        depth = 0
        for i in range(end - 1, stop, -1):
            c = text[i]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    return i
        raise RuntimeError("unmatched brace walking back from JS dict")

    name_start = _brace_start(name_end)
    hash_start = _brace_start(hash_end, stop=name_end)

    pair_re = re.compile(r'(\d+):"([^"]+)"')
    name_dict = dict(pair_re.findall(text[name_start:name_end]))
    hash_dict = dict(pair_re.findall(text[hash_start:hash_end]))
    for k, name in name_dict.items():
        h = hash_dict.get(k)
        if h is None:
            continue
        yield _xclid.script_url(name, f"{h}a")


_xclid.get_scripts_list = _patched_get_scripts_list


def _to_account(u) -> AccountRecord:
    return AccountRecord(
        user_id=str(u.id),
        handle=u.username,
        display_name=u.displayname or "",
        description=u.rawDescription or "",
        followers_count=u.followersCount or 0,
        following_count=u.friendsCount or 0,
        tweet_count=u.statusesCount or 0,
        created_at=u.created.isoformat() if u.created else None,
        verified=getattr(u, "verified", None),
    )


def _to_tweet(t) -> TweetRecord:
    mentions = getattr(t, "mentionedUsers", []) or []
    return TweetRecord(
        tweet_id=str(t.id),
        user_id=str(t.user.id),
        handle=t.user.username,
        created_at=t.date.isoformat() if t.date else "",
        text=t.rawContent or "",
        lang=getattr(t, "lang", None),
        retweet_count=t.retweetCount or 0,
        like_count=t.likeCount or 0,
        reply_count=t.replyCount or 0,
        quote_count=t.quoteCount or 0,
        is_retweet=bool(getattr(t, "retweetedTweet", None)),
        is_reply=t.inReplyToTweetId is not None,
        in_reply_to_user_id=str(t.inReplyToUser.id) if t.inReplyToUser else None,
        mentioned_user_ids=[str(m.id) for m in mentions],
        mentioned_handles=[m.username for m in mentions],
    )


async def get_user(api: API, handle: str) -> AccountRecord | None:
    u = await api.user_by_login(handle)
    return _to_account(u) if u else None


async def iter_followers(api: API, user_id: int, limit: int | None = None) -> AsyncIterator[AccountRecord]:
    async for u in api.followers(user_id, limit=limit or -1):
        yield _to_account(u)


async def list_followers(api: API, handle: str, limit: int | None = None) -> list[AccountRecord]:
    target = await api.user_by_login(handle)
    if target is None:
        raise RuntimeError(f"user @{handle} not found")
    users = await gather(api.followers(target.id, limit=limit or -1))
    return [_to_account(u) for u in users]


async def fetch_tweets(
    api: API,
    user_id: int,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
) -> list[TweetRecord]:
    """
    Pull a user's tweets. twscrape returns newest-first; we filter by date
    client-side because the per-user timeline endpoint doesn't take a range.
    """
    tweets = await gather(api.user_tweets_and_replies(user_id, limit=limit or -1))
    out: list[TweetRecord] = []
    for t in tweets:
        if since and t.date and t.date < since:
            continue
        if until and t.date and t.date > until:
            continue
        out.append(_to_tweet(t))
    return out
