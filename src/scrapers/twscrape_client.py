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

from twscrape import API, gather

from .base import AccountRecord, TweetRecord


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
