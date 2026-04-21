"""
Base scraper abstraction.

The paper used the Twitter v1.1 API (pre-Musk, 2021). That access tier is
effectively gone. Two realistic options today:

1. `twscrape` — scrapes the web UI via logged-in X accounts you provide.
   Best match for reproducing the paper without paying. Requires real
   X accounts (test/throwaway accounts work but must be warmed up).
2. `tweepy` against the v2 API — requires a paid X API tier. Stable but
   costly at the volume this paper used.

The scrapers below default to twscrape. Tweepy is stubbed so it's easy
to swap if/when the user has API credentials.
"""
from dataclasses import dataclass


@dataclass
class AccountRecord:
    user_id: str
    handle: str
    display_name: str
    description: str
    followers_count: int
    following_count: int
    tweet_count: int
    created_at: str | None
    verified: bool | None = None


@dataclass
class TweetRecord:
    tweet_id: str
    user_id: str
    handle: str
    created_at: str
    text: str
    lang: str | None
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    is_retweet: bool
    is_reply: bool
    in_reply_to_user_id: str | None
    mentioned_user_ids: list[str]
    mentioned_handles: list[str]
