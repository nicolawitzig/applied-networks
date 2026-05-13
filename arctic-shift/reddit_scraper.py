"""
Shared Arctic Shift API client — imported by scrape_user_subreddits_crossref.py
and other scrapers that need the base HTTP client and dataclasses.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class RedditPost:
    """One post record returned by the Arctic Shift posts/search endpoint."""
    id: str
    title: str
    author: str
    subreddit: str
    created_utc: int
    score: int
    num_comments: int
    selftext: str
    url: str
    retrieved_on: int

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "RedditPost":
        """Construct from raw API response dict."""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            author=data.get("author", ""),
            subreddit=data.get("subreddit", ""),
            created_utc=data.get("created_utc", 0),
            score=data.get("score", 0),
            num_comments=data.get("num_comments", 0),
            selftext=data.get("selftext", ""),
            url=data.get("url", ""),
            retrieved_on=data.get("retrieved_on", 0),
        )


@dataclass
class RedditComment:
    """One comment record returned by the Arctic Shift comments/search endpoint."""
    id: str
    body: str
    author: str
    subreddit: str
    created_utc: int
    score: int
    link_id: str      # t3_<post_id> — the thread this comment belongs to
    parent_id: str    # t3_<post_id> for top-level, t1_<comment_id> for replies
    retrieved_on: int

    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> "RedditComment":
        """Construct from raw API response dict."""
        return cls(
            id=data.get("id", ""),
            body=data.get("body", ""),
            author=data.get("author", ""),
            subreddit=data.get("subreddit", ""),
            created_utc=data.get("created_utc", 0),
            score=data.get("score", 0),
            link_id=data.get("link_id", ""),
            parent_id=data.get("parent_id", ""),
            retrieved_on=data.get("retrieved_on", 0),
        )


class ArcticShiftScraper:
    """HTTP client for the Arctic Shift Reddit archive API."""

    BASE_URL         = "https://arctic-shift.photon-reddit.com"
    RATE_LIMIT_DELAY = 1.0    # minimum seconds between any two requests

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AcademicResearchBot/1.0",
            "Accept": "application/json",
        })
        self.last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Block until at least RATE_LIMIT_DELAY seconds have passed since the last request."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def search_posts(
        self,
        subreddit: str,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None,
        sort: str = "desc",
    ) -> List[RedditPost]:
        """Return up to limit posts from subreddit in the given date window."""
        params: Dict[str, Any] = {"subreddit": subreddit, "limit": min(limit, 100), "sort": sort}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        self._rate_limit()
        resp = self.session.get(f"{self.BASE_URL}/api/posts/search", params=params, timeout=30)
        if resp.status_code == 429:
            print("Rate limited — waiting 60 s")
            time.sleep(60)
            return self.search_posts(subreddit, limit, after, before, sort)
        resp.raise_for_status()
        return [RedditPost.from_api_data(p) for p in resp.json().get("data", [])]

    def search_comments(
        self,
        subreddit: str,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None,
        sort: str = "desc",
    ) -> List[RedditComment]:
        """Return up to limit comments from subreddit in the given date window."""
        params: Dict[str, Any] = {"subreddit": subreddit, "limit": min(limit, 100), "sort": sort}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        self._rate_limit()
        resp = self.session.get(f"{self.BASE_URL}/api/comments/search", params=params, timeout=30)
        if resp.status_code == 429:
            print("Rate limited — waiting 60 s")
            time.sleep(60)
            return self.search_comments(subreddit, limit, after, before, sort)
        resp.raise_for_status()
        return [RedditComment.from_api_data(c) for c in resp.json().get("data", [])]
