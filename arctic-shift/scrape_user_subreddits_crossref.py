#!/usr/bin/env python3
"""
Reddit User Cross-Subreddit Analysis (Simplified)
For a target subreddit + date range, discover ALL users via exhaustive pagination,
then find what other subreddits each user interacts in.

Output: single CSV with packed subreddit:count format.

Usage:
  python scrape_user_subreddits_crossref.py --subreddit ethz --after 2024-01-01 --before 2026-01-01
"""

import sys
import time
import csv
import argparse
from datetime import datetime
from typing import Dict, Optional, Set, List, Tuple
from dataclasses import dataclass
from reddit_scraper import ArcticShiftScraper, RedditPost, RedditComment


@dataclass
class UserActivity:
    """Tracks a user's activity across all subreddits (posts + comments combined)"""
    username: str
    subreddits: Dict[str, int]
    total_posts: int = 0
    total_comments: int = 0
    first_date: Optional[int] = None
    last_date: Optional[int] = None

    @property
    def total_items(self) -> int:
        return self.total_posts + self.total_comments

    def add_post(self, p: RedditPost) -> None:
        self.total_posts += 1
        self.subreddits[p.subreddit] = self.subreddits.get(p.subreddit, 0) + 1
        if self.first_date is None or p.created_utc < self.first_date:
            self.first_date = p.created_utc
        if self.last_date is None or p.created_utc > self.last_date:
            self.last_date = p.created_utc

    def add_comment(self, c: RedditComment) -> None:
        self.total_comments += 1
        self.subreddits[c.subreddit] = self.subreddits.get(c.subreddit, 0) + 1
        if self.first_date is None or c.created_utc < self.first_date:
            self.first_date = c.created_utc
        if self.last_date is None or c.created_utc > self.last_date:
            self.last_date = c.created_utc


class CrossSubredditAnalyzer(ArcticShiftScraper):
    """Paginates exhaustively through a subreddit, then cross-references each user."""

    def _paginate_posts(self, subreddit: str, after: Optional[str],
                        before: Optional[str]) -> List[RedditPost]:
        """Fetch ALL posts from a subreddit in the date range via pagination."""
        all_posts: List[RedditPost] = []
        before_param = before
        limit = 100

        while True:
            batch = self.search_posts(subreddit, limit=limit, after=after,
                                      before=before_param, sort='desc')
            if not batch:
                break
            all_posts.extend(batch)
            batch_len = len(batch)
            print(f"  Fetched {batch_len} posts (total: {len(all_posts)})", flush=True)
            oldest = min(batch, key=lambda p: p.created_utc)
            before_param = str(oldest.created_utc)
            if batch_len < limit:
                break
            time.sleep(1.2)
        return all_posts

    def _paginate_comments(self, subreddit: str, after: Optional[str],
                           before: Optional[str]) -> List[RedditComment]:
        """Fetch ALL comments from a subreddit in the date range via pagination."""
        all_comments: List[RedditComment] = []
        before_param = before
        limit = 100

        while True:
            batch = self.search_comments(subreddit, limit=limit, after=after,
                                         before=before_param, sort='desc')
            if not batch:
                break
            all_comments.extend(batch)
            batch_len = len(batch)
            print(f"  Fetched {batch_len} comments (total: {len(all_comments)})", flush=True)
            oldest = min(batch, key=lambda c: c.created_utc)
            before_param = str(oldest.created_utc)
            if batch_len < limit:
                break
            time.sleep(1.2)
        return all_comments

    def _paginate_user_posts(self, username: str, after: Optional[str],
                             before: Optional[str]) -> List[RedditPost]:
        """Fetch ALL posts by a user in the date range."""
        all_items: List[RedditPost] = []
        before_param = before
        limit = 100

        while True:
            params: Dict = {'author': username, 'limit': limit, 'sort': 'desc'}
            if after:
                params['after'] = after
            if before_param:
                params['before'] = before_param

            self._rate_limit()
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/api/posts/search",
                    params=params, timeout=30
                )
                resp.raise_for_status()
                items = [RedditPost.from_api_data(p) for p in resp.json().get('data', [])]
            except Exception as e:
                print(f"  Error fetching posts for u/{username}: {e}")
                break

            if not items:
                break
            all_items.extend(items)
            oldest = min(items, key=lambda x: x.created_utc)
            before_param = str(oldest.created_utc)
            if len(items) < limit:
                break
            time.sleep(1.2)
        return all_items

    def _paginate_user_comments(self, username: str, after: Optional[str],
                                before: Optional[str]) -> List[RedditComment]:
        """Fetch ALL comments by a user in the date range."""
        all_items: List[RedditComment] = []
        before_param = before
        limit = 100

        while True:
            params: Dict = {'author': username, 'limit': limit, 'sort': 'desc'}
            if after:
                params['after'] = after
            if before_param:
                params['before'] = before_param

            self._rate_limit()
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/api/comments/search",
                    params=params, timeout=30
                )
                resp.raise_for_status()
                items = [RedditComment.from_api_data(c) for c in resp.json().get('data', [])]
            except Exception as e:
                print(f"  Error fetching comments for u/{username}: {e}")
                break

            if not items:
                break
            all_items.extend(items)
            oldest = min(items, key=lambda x: x.created_utc)
            before_param = str(oldest.created_utc)
            if len(items) < limit:
                break
            time.sleep(1.2)
        return all_items

    def discover_users(self, subreddit: str, after: Optional[str],
                       before: Optional[str]) -> Tuple[Set[str], int, int]:
        """Discover all unique authors from posts and comments in a subreddit."""
        print(f"Step 1a: Fetching all posts in r/{subreddit}...")
        posts = self._paginate_posts(subreddit, after, before)
        users: Set[str] = set()
        for p in posts:
            if p.author and p.author != '[deleted]':
                users.add(p.author)
        print(f"  Found {len(posts)} posts from {len(users)} unique users")

        print(f"Step 1b: Fetching all comments in r/{subreddit}...")
        comments = self._paginate_comments(subreddit, after, before)
        comment_users: Set[str] = set()
        for c in comments:
            if c.author and c.author != '[deleted]':
                comment_users.add(c.author)
        before_count = len(users)
        users |= comment_users
        new_users = len(users) - before_count
        print(f"  Found {len(comments)} comments from {len(comment_users)} unique users ({new_users} new)")

        print(f"Total unique users: {len(users)}")
        return users, len(posts), len(comments)

    def analyze_user(self, username: str, after: Optional[str],
                     before: Optional[str]) -> UserActivity:
        """Get all posts and comments for a user in the date range."""
        posts = self._paginate_user_posts(username, after, before)
        comments = self._paginate_user_comments(username, after, before)

        activity = UserActivity(username=username, subreddits={})
        for p in posts:
            activity.add_post(p)
        for c in comments:
            activity.add_comment(c)
        return activity


def format_ts(ts: Optional[int]) -> str:
    if ts is None or ts == 0:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


def save_csv(activities: Dict[str, UserActivity], filename: str) -> None:
    """Save to CSV with packed subreddit:count format."""
    rows = []
    for act in activities.values():
        rows.append({
            'username': act.username,
            'total_posts': act.total_posts,
            'total_comments': act.total_comments,
            'total_items': act.total_items,
            'subreddit_count': len(act.subreddits),
            'first_date': format_ts(act.first_date),
            'last_date': format_ts(act.last_date),
            'subreddits': ', '.join(f"{sub}:{cnt}" for sub, cnt in
                                    sorted(act.subreddits.items(), key=lambda x: -x[1]))
        })

    fieldnames = ['username', 'total_posts', 'total_comments', 'total_items',
                  'subreddit_count', 'first_date', 'last_date', 'subreddits']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(activities)} users to {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Discover all users in a subreddit + date range and find their cross-subreddit activity.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subreddit ethz --after 2024-01-01 --before 2026-01-01
  %(prog)s --subreddit askswitzerland --after 2025-01-01 --before 2025-06-01 --output results.csv
        """
    )
    parser.add_argument('--subreddit', required=True, help='Target subreddit (without r/)')
    parser.add_argument('--after', required=True, help='Start date YYYY-MM-DD (or epoch)')
    parser.add_argument('--before', required=True, help='End date YYYY-MM-DD (or epoch)')
    parser.add_argument('--output', help='Output CSV filename (default: <subreddit>_users_crossref.csv)')
    args = parser.parse_args()

    output = args.output or f"{args.subreddit}_users_crossref.csv"

    analyzer = CrossSubredditAnalyzer()
    users, total_posts, total_comments = analyzer.discover_users(
        args.subreddit, args.after, args.before
    )

    if not users:
        print("No users found in the given date range.")
        return 1

    print(f"\nStep 2: Cross-referencing {len(users)} users...")
    activities: Dict[str, UserActivity] = {}
    for i, username in enumerate(users, 1):
        print(f"  [{i}/{len(users)}] u/{username}...", end="", flush=True)
        act = analyzer.analyze_user(username, args.after, args.before)
        if act.total_items > 0:
            activities[username] = act
            print(f" {act.total_items} items in {len(act.subreddits)} subreddits")
        else:
            print(" no activity found")
        time.sleep(0.5)

    save_csv(activities, output)

    active = len(activities)
    print(f"\nSummary: {total_posts} posts, {total_comments} comments in r/{args.subreddit}")
    print(f"  {len(users)} unique users discovered, {active} with cross-subreddit activity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
