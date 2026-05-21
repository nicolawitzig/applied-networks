#!/usr/bin/env python3
"""
Fetch the actual post and comment text for users listed in subreddit crossref CSVs.

For each username, writes one JSON file to results/<username>.json containing
their posts and comments. Skips users whose file already exists (safe to resume).

Usage:
  python scrape_user_posts.py --input subreddit_scrapes/ethz_users_scrape_2024_Jan_Jun.csv
  python scrape_user_posts.py --input subreddit_scrapes/ethz_users_scrape_2024_Jul_Dec.csv --subreddit ethz
  python scrape_user_posts.py --input subreddit_scrapes/ethz_users_scrape_2025_Dev_Jun.csv --subreddit all
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import threading
from collections import defaultdict
import concurrent.futures
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

BATCH_SIZE = 100  # Arctic Shift API maximum items per request
BATCH_DELAY = 0.0  # seconds between pagination requests within one user
USER_DELAY = 0.0  # additional pause between users (on top of rate limiting)

SKIP_AUTHORS = {
    "[deleted]",
    "AutoModerator",
    "",
}  # authors whose content carries no signal


def is_bot(username: str) -> bool:
    """Heuristic: matches common Reddit bot naming patterns (case-insensitive)."""
    u = username.lower()
    return u.endswith("bot") or "_bot" in u or u.startswith("bot_")


# ---------------------------------------------------------------------------
# Arctic Shift API client
# ---------------------------------------------------------------------------


@dataclass
class RedditPost:
    """One post record from the Arctic Shift posts/search endpoint."""

    id: str
    title: str
    author: str
    subreddit: str
    created_utc: int
    score: int
    num_comments: int
    selftext: str
    url: str

    @classmethod
    def from_api_data(cls, d: Dict[str, Any]) -> "RedditPost":
        """Construct from raw API response dict."""
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            author=d.get("author", ""),
            subreddit=d.get("subreddit", ""),
            created_utc=d.get("created_utc", 0),
            score=d.get("score", 0),
            num_comments=d.get("num_comments", 0),
            selftext=d.get("selftext", ""),
            url=d.get("url", ""),
        )


@dataclass
class RedditComment:
    """One comment record from the Arctic Shift comments/search endpoint."""

    id: str
    body: str
    author: str
    subreddit: str
    created_utc: int
    score: int
    link_id: str  # t3_<post_id> — thread this comment belongs to
    parent_id: str  # t3_<post_id> or t1_<comment_id>

    @classmethod
    def from_api_data(cls, d: Dict[str, Any]) -> "RedditComment":
        """Construct from raw API response dict."""
        return cls(
            id=d.get("id", ""),
            body=d.get("body", ""),
            author=d.get("author", ""),
            subreddit=d.get("subreddit", ""),
            created_utc=d.get("created_utc", 0),
            score=d.get("score", 0),
            link_id=d.get("link_id", ""),
            parent_id=d.get("parent_id", ""),
        )


class ArcticShiftClient:
    """Minimal HTTP client for the Arctic Shift Reddit archive API."""

    BASE_URL = "https://arctic-shift.photon-reddit.com"
    MIN_DELAY = 0.0  # minimum seconds between any two requests

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "AcademicResearchBot/1.0",
                "Accept": "application/json",
            }
        )
        self._last_request: float = 0.0
        self._lock = threading.Lock()
        self.rate_limit_hits = 0
        self.rate_limit_lock = threading.Lock()

    def _rate_limit(self) -> None:
        """Block until MIN_DELAY seconds have elapsed since the last request."""
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self.MIN_DELAY:
                time.sleep(self.MIN_DELAY - elapsed)
            self._last_request = time.time()

    MAX_RETRIES = 4  # max attempts per request before giving up
    RETRY_DELAY = 45.0  # seconds to wait after a timeout before retrying

    def _get(
        self, endpoint: str, params: Dict[str, Any], _attempt: int = 1
    ) -> List[dict]:
        """GET one page from the API; retries on 429 and read timeouts."""
        self._rate_limit()
        try:
            resp = self.session.get(
                f"{self.BASE_URL}{endpoint}", params=params, timeout=60
            )
        except requests.exceptions.ReadTimeout:
            if _attempt >= self.MAX_RETRIES:
                raise
            wait = self.RETRY_DELAY * _attempt   # back off linearly
            time.sleep(wait)
            return self._get(endpoint, params, _attempt + 1)
        if resp.status_code == 429:
            with self.rate_limit_lock:
                self.rate_limit_hits += 1
            time.sleep(60)
            return self._get(endpoint, params, _attempt)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def fetch_user_posts(
        self,
        username: str,
        subreddit: Optional[str],
        after: Optional[str],
        before: Optional[str],
    ) -> List[RedditPost]:
        """Paginate all posts by username; optionally scoped to one subreddit and date range."""
        all_items: List[RedditPost] = []
        seen: Set[str] = set()  # guards against duplicate IDs at cursor boundary
        before_cursor = before

        while True:
            params: Dict[str, Any] = {
                "author": username,
                "limit": BATCH_SIZE,
                "sort": "desc",
            }
            if subreddit:
                params["subreddit"] = subreddit
            if after:
                params["after"] = after
            if before_cursor:
                params["before"] = before_cursor

            raw = self._get("/api/posts/search", params)
            if not raw:
                break
            new = [RedditPost.from_api_data(d) for d in raw if d.get("id") not in seen]
            seen.update(p.id for p in new)
            all_items.extend(new)
            if len(raw) < BATCH_SIZE:
                break
            before_cursor = str(min(new, key=lambda p: p.created_utc).created_utc)
            time.sleep(BATCH_DELAY)

        return all_items

    def fetch_user_comments(
        self,
        username: str,
        subreddit: Optional[str],
        after: Optional[str],
        before: Optional[str],
    ) -> List[RedditComment]:
        """Paginate all comments by username; optionally scoped to one subreddit and date range."""
        all_items: List[RedditComment] = []
        seen: Set[str] = set()
        before_cursor = before

        while True:
            params: Dict[str, Any] = {
                "author": username,
                "limit": BATCH_SIZE,
                "sort": "desc",
            }
            if subreddit:
                params["subreddit"] = subreddit
            if after:
                params["after"] = after
            if before_cursor:
                params["before"] = before_cursor

            raw = self._get("/api/comments/search", params)
            if not raw:
                break
            new = [
                RedditComment.from_api_data(d) for d in raw if d.get("id") not in seen
            ]
            seen.update(c.id for c in new)
            all_items.extend(new)
            if len(raw) < BATCH_SIZE:
                break
            before_cursor = str(min(new, key=lambda c: c.created_utc).created_utc)
            time.sleep(BATCH_DELAY)

        return all_items


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

MONTH_ABBR: Dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "dev": 1,  # typo in ethz_users_scrape_2025_Dev_Jun.csv — treat as Jan
}


def parse_date_range(csv_path: str) -> Tuple[str, str]:
    """
    Derive after/before dates from the filename pattern *_YYYY_Mon_Mon.csv.
    Returns (after, before) as YYYY-MM-DD strings.
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    match = re.search(r"_(\d{4})_([A-Za-z]+)_([A-Za-z]+)$", stem)
    assert match, f"Cannot parse date range from filename: {stem!r}"
    year = int(match.group(1))
    start_mon = MONTH_ABBR[match.group(2).lower()]
    end_mon = MONTH_ABBR[match.group(3).lower()]
    last_day = calendar.monthrange(year, end_mon)[1]
    return f"{year}-{start_mon:02d}-01", f"{year}-{end_mon:02d}-{last_day:02d}"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_usernames(csv_paths: List[str]) -> List[str]:
    """
    Read and deduplicate usernames from one or more crossref CSVs.
    Reads only the first column per line to avoid parsing the large packed subreddits field.
    """
    seen: Set[str] = set()
    ordered: List[str] = []  # stable order for reproducible progress
    for path in csv_paths:
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            assert header[0] == "username", (
                f"Expected first column 'username', got {header[0]!r}"
            )
            for line in f:
                u = line.split(",", 1)[0].strip()
                if u and u not in SKIP_AUTHORS and not is_bot(u) and u not in seen:
                    seen.add(u)
                    ordered.append(u)
    return ordered


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def load_existing_user(out_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load existing posts and comments from a user JSON; returns ([], []) if file absent."""
    if not os.path.exists(out_path):
        return [], []
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("posts", []), data.get("comments", [])


def merge_and_save(
    username: str,
    out_path: str,
    existing_posts: List[Dict],
    existing_comments: List[Dict],
    fetched_posts: List[Any],
    fetched_comments: List[Any],
) -> Tuple[int, int]:
    """
    Append only posts/comments whose IDs are not already in the existing file.
    Returns (n_new_posts, n_new_comments) added.
    """
    seen_post_ids: Set[str] = {p["id"] for p in existing_posts}
    seen_comment_ids: Set[str] = {c["id"] for c in existing_comments}

    new_posts = [asdict(p) for p in fetched_posts if p.id not in seen_post_ids]
    new_comments = [asdict(c) for c in fetched_comments if c.id not in seen_comment_ids]

    payload = {
        "username": username,
        "posts": existing_posts + new_posts,
        "comments": existing_comments + new_comments,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return len(new_posts), len(new_comments)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch posts and comments for users from a directory of subreddit crossref CSVs; saves one JSON per user.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir subreddit_scrapes
  %(prog)s --input-dir subreddit_scrapes --subreddit ethz
        """,
    )
    parser.add_argument("--input-dir",  default="subreddit_scrapes",
                        help="Directory containing crossref CSV files (date range is parsed from the filenames)")
    parser.add_argument("--year",       type=int, default=None,
                        help="Optional. If set (e.g. 2024), ignores the filename dates and fetches everything for the given year.")
    parser.add_argument("--subreddit",  default="all",
                        help="Restrict to this subreddit. Pass 'all' for no filter. (default: all)")
    parser.add_argument("--output-dir", default="results/user_posts",
                        help="Directory to write per-user JSON files (default: results/user_posts/)")
    parser.add_argument("--workers",    type=int, default=100,
                        help="Number of concurrent threads to use (default: 100)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sub_filter: Optional[str] = None if args.subreddit == "all" else args.subreddit
    
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return 1

    client = ArcticShiftClient()
    user_locks = defaultdict(threading.Lock)
    tasks = []

    def process_user(username, after, before, out_path):
        with user_locks[username]:
            existing_posts, existing_comments = load_existing_user(out_path)
            try:
                posts    = client.fetch_user_posts(username, sub_filter, after, before)
                comments = client.fetch_user_comments(username, sub_filter, after, before)
            except requests.exceptions.RequestException as exc:
                return f"[{username}] SKIP (network error: {exc})"

            n_new_posts, n_new_comments = merge_and_save(
                username, out_path, existing_posts, existing_comments, posts, comments
            )
            return f"[{username}] {n_new_posts} new posts, {n_new_comments} new comments (had {len(existing_posts)} posts, {len(existing_comments)} comments)"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for csv_path in sorted(csv_files):
            if args.year:
                after = f"{args.year}-01-01"
                before = f"{args.year}-12-31"
            else:
                after, before = parse_date_range(csv_path)
            print(f"Queuing users from {csv_path} (Date range: {after} → {before})")
            
            usernames = load_usernames([csv_path])
            for username in usernames:
                out_path = os.path.join(args.output_dir, f"{username}.json")
                future = executor.submit(process_user, username, after, before, out_path)
                tasks.append(future)
        
        total = len(tasks)
        print(f"\nProcessing {total} user tasks with {args.workers} concurrent workers...")
        completed = 0
        
        # Initial empty progress bar
        sys.stdout.write(f"\r[0/{total}] [{' ' * 40}] 0.0% | Rate limits hit: 0")
        sys.stdout.flush()

        for future in concurrent.futures.as_completed(tasks):
            completed += 1
            future.result() # consume the result but don't print it
            
            # Update progress bar
            percent = completed / total
            filled = int(40 * percent)
            bar = '=' * filled + '-' * (40 - filled)
            sys.stdout.write(f"\r[{completed}/{total}] [{bar}] {percent*100:.1f}% | Rate limits hit: {client.rate_limit_hits}")
            sys.stdout.flush()

    print(f"\n\nDone. Files written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
