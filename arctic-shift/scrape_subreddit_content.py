#!/usr/bin/env python3
"""
Scrape ALL posts, comments, and replies from a subreddit and assemble into threads.

Outputs:
  results/{sub}_threads.jsonl  — one JSON thread per line, nested structure (for LLM training)
  results/{sub}_flat.csv       — flat table, one row per post/comment (for classification)

Usage:
  python scrape_subreddit_content.py --subreddit ethz --after 2024-01-01 --before 2026-01-01
  python scrape_subreddit_content.py --subreddit ethz --after 2024-01-01 --before 2026-01-01 --output-dir results/
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from reddit_scraper import ArcticShiftScraper, RedditComment, RedditPost

BATCH_DELAY = 1.2    # seconds between pagination requests — stays within API rate limits
BATCH_SIZE = 100     # Arctic Shift API maximum per request


def _ts_to_date(ts: int) -> str:
    """Unix timestamp → ISO date string used in all output rows."""
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def _paginate_posts(
    scraper: ArcticShiftScraper,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
) -> List[RedditPost]:
    """Exhaust all pages of posts in the date window; deduplicates across pages."""
    all_posts: List[RedditPost] = []
    seen: set = set()        # post IDs already collected — guards against cursor-timestamp ties
    before_cursor = before

    while True:
        batch = scraper.search_posts(
            subreddit, limit=BATCH_SIZE, after=after, before=before_cursor, sort="desc"
        )
        if not batch:
            break
        new = [p for p in batch if p.id not in seen]
        seen.update(p.id for p in new)
        all_posts.extend(new)
        print(f"  posts fetched: {len(all_posts)}", end="\r", flush=True)
        before_cursor = str(min(batch, key=lambda p: p.created_utc).created_utc)
        if len(batch) < BATCH_SIZE:
            break
        time.sleep(BATCH_DELAY)

    print(f"  posts fetched: {len(all_posts)}")
    return all_posts


def _paginate_comments(
    scraper: ArcticShiftScraper,
    subreddit: str,
    after: Optional[str],
    before: Optional[str],
) -> List[RedditComment]:
    """Exhaust all pages of comments in the date window; deduplicates across pages."""
    all_comments: List[RedditComment] = []
    seen: set = set()        # comment IDs already collected
    before_cursor = before

    while True:
        batch = scraper.search_comments(
            subreddit, limit=BATCH_SIZE, after=after, before=before_cursor, sort="desc"
        )
        if not batch:
            break
        new = [c for c in batch if c.id not in seen]
        seen.update(c.id for c in new)
        all_comments.extend(new)
        print(f"  comments fetched: {len(all_comments)}", end="\r", flush=True)
        before_cursor = str(min(batch, key=lambda c: c.created_utc).created_utc)
        if len(batch) < BATCH_SIZE:
            break
        time.sleep(BATCH_DELAY)

    print(f"  comments fetched: {len(all_comments)}")
    return all_comments


def _build_comment_node(
    comment: RedditComment,
    children_map: Dict[str, List[str]],   # bare parent ID → list of child comment IDs
    comments_by_id: Dict[str, RedditComment],
    depth: int,
) -> dict:
    """Recursively build a nested comment dict; depth tracks nesting level for the flat CSV."""
    node = {
        "id": comment.id,
        "parent_id": comment.parent_id,    # original Reddit parent_id (t3_/t1_ prefixed)
        "author": comment.author,
        "created_utc": comment.created_utc,
        "date": _ts_to_date(comment.created_utc),
        "score": comment.score,
        "body": comment.body,
        "depth": depth,
        "replies": [],
    }
    for child_id in children_map.get(comment.id, []):
        if child_id in comments_by_id:
            node["replies"].append(
                _build_comment_node(comments_by_id[child_id], children_map, comments_by_id, depth + 1)
            )
    return node


def _assemble_threads(
    posts: List[RedditPost],
    comments: List[RedditComment],
) -> List[dict]:
    """
    Join posts and comments into nested thread objects.
    Comments outside the scraped post set are attached to a synthetic thread so no text is lost.
    """
    comments_by_id: Dict[str, RedditComment] = {c.id: c for c in comments}
    post_ids: set = {p.id for p in posts}

    # bare parent ID (strip t3_/t1_ prefix) → child comment IDs
    children_map: Dict[str, List[str]] = {}
    for c in comments:
        bare_parent = c.parent_id.split("_", 1)[1] if "_" in c.parent_id else c.parent_id
        children_map.setdefault(bare_parent, []).append(c.id)

    threads: List[dict] = []
    for post in sorted(posts, key=lambda p: p.created_utc):
        top_level = [    # direct children of the post — depth=0
            c for c in comments
            if c.link_id == f"t3_{post.id}" and c.parent_id == f"t3_{post.id}"
        ]
        threads.append({
            "id": post.id,
            "title": post.title,
            "author": post.author,
            "created_utc": post.created_utc,
            "date": _ts_to_date(post.created_utc),
            "score": post.score,
            "num_comments": post.num_comments,
            "selftext": post.selftext,
            "url": post.url,
            "comments": [
                _build_comment_node(c, children_map, comments_by_id, depth=0)
                for c in sorted(top_level, key=lambda c: c.created_utc)
            ],
        })

    # Orphaned comments (their parent post is outside the date window) → synthetic thread
    orphaned = [c for c in comments if c.link_id.split("_", 1)[1] not in post_ids]
    if orphaned:
        print(f"  note: {len(orphaned)} comments link to posts outside date range — grouped in _orphaned thread")
        threads.append({
            "id": "_orphaned",
            "title": "[parent posts outside scraped date range]",
            "author": "",
            "created_utc": 0,
            "date": "",
            "score": 0,
            "num_comments": len(orphaned),
            "selftext": "",
            "url": "",
            "comments": [
                _build_comment_node(c, children_map, comments_by_id, depth=0)
                for c in sorted(orphaned, key=lambda c: c.created_utc)
            ],
        })

    return threads


def _flatten_comments(comments: list, post_id: str, rows: list) -> None:
    """Recursively walk comment nodes and append one flat CSV row per comment."""
    for node in comments:
        rows.append({
            "type": "comment",
            "id": node["id"],
            "post_id": post_id,
            "parent_id": node["parent_id"],
            "depth": node["depth"],
            "author": node["author"],
            "date": node["date"],
            "score": node["score"],
            "text": node["body"],
        })
        _flatten_comments(node["replies"], post_id, rows)


def _write_jsonl(threads: List[dict], path: str) -> None:
    """Write one JSON object per line — standard JSONL format expected by most LLM training pipelines."""
    with open(path, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")
    print(f"Saved {len(threads)} threads → {path}")


def _write_flat_csv(threads: List[dict], path: str) -> None:
    """Write every post and comment as a flat row — the input format for the classification script."""
    fieldnames = ["type", "id", "post_id", "parent_id", "depth", "author", "date", "score", "text"]
    rows: list = []
    for thread in threads:
        if thread["id"] != "_orphaned":
            rows.append({
                "type": "post",
                "id": thread["id"],
                "post_id": thread["id"],
                "parent_id": "",
                "depth": 0,
                "author": thread["author"],
                "date": thread["date"],
                "score": thread["score"],
                "text": (thread["title"] + "\n\n" + thread["selftext"]).strip(),
            })
        _flatten_comments(thread["comments"], thread["id"], rows)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_posts = sum(1 for r in rows if r["type"] == "post")
    n_comments = sum(1 for r in rows if r["type"] == "comment")
    print(f"Saved {n_posts} posts + {n_comments} comments → {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape all posts and comments from a subreddit into thread-structured output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subreddit ethz --after 2024-01-01 --before 2026-01-01
  %(prog)s --subreddit ethz --after 2024-01-01 --before 2026-01-01 --output-dir results/
        """,
    )
    parser.add_argument("--subreddit", required=True, help="Target subreddit (without r/)")
    parser.add_argument("--after",     required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--before",    required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results/)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sub = args.subreddit
    scraper = ArcticShiftScraper()

    print(f"Fetching all posts from r/{sub}  ({args.after} → {args.before})...")
    posts = _paginate_posts(scraper, sub, args.after, args.before)
    assert posts, f"No posts found in r/{sub} for the given date range — check subreddit name and dates"

    print(f"\nFetching all comments from r/{sub}...")
    comments = _paginate_comments(scraper, sub, args.after, args.before)

    print(f"\nAssembling {len(posts)} posts + {len(comments)} comments into threads...")
    threads = _assemble_threads(posts, comments)

    stem = f"{sub}_{args.after}_{args.before}"    # e.g. ethz_2024-01-01_2026-01-01
    _write_jsonl(threads, os.path.join(args.output_dir, f"{stem}_threads.jsonl"))
    _write_flat_csv(threads, os.path.join(args.output_dir, f"{stem}_flat.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
