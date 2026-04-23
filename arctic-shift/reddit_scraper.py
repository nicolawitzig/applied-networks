#!/usr/bin/env python3
"""
Reddit Subreddit Scraper using Arctic Shift API
Academic research tool for analyzing university social media presence
"""

import requests
import json
import csv
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import sys


@dataclass
class RedditPost:
    """Data class representing a Reddit post from Arctic Shift API"""
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
    def from_api_data(cls, data: Dict[str, Any]) -> 'RedditPost':
        """Create RedditPost from API response data"""
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            author=data.get('author', ''),
            subreddit=data.get('subreddit', ''),
            created_utc=data.get('created_utc', 0),
            score=data.get('score', 0),
            num_comments=data.get('num_comments', 0),
            selftext=data.get('selftext', ''),
            url=data.get('url', ''),
            retrieved_on=data.get('retrieved_on', 0)
        )


@dataclass
class RedditComment:
    """Data class representing a Reddit comment from Arctic Shift API"""
    id: str
    body: str
    author: str
    subreddit: str
    created_utc: int
    score: int
    link_id: str
    parent_id: str
    retrieved_on: int
    
    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> 'RedditComment':
        """Create RedditComment from API response data"""
        return cls(
            id=data.get('id', ''),
            body=data.get('body', ''),
            author=data.get('author', ''),
            subreddit=data.get('subreddit', ''),
            created_utc=data.get('created_utc', 0),
            score=data.get('score', 0),
            link_id=data.get('link_id', ''),
            parent_id=data.get('parent_id', ''),
            retrieved_on=data.get('retrieved_on', 0)
        )


class ArcticShiftScraper:
    """Client for querying Reddit data from Arctic Shift API"""
    
    BASE_URL = "https://arctic-shift.photon-reddit.com"
    RATE_LIMIT_DELAY = 1.0  # seconds between requests
    
    def __init__(self, user_agent: str = "AcademicResearchBot/1.0"):
        """Initialize scraper with user agent for identification"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json'
        })
        self.last_request_time = 0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def search_posts(self, subreddit: str, limit: int = 100, 
                    after: Optional[str] = None,
                    before: Optional[str] = None,
                    sort: str = 'desc') -> List[RedditPost]:
        """
        Search for posts in a subreddit using Arctic Shift API
        
        Args:
            subreddit: Subreddit name (without r/)
            limit: Number of posts to retrieve (1-100)
            after: Start date (ISO format or epoch)
            before: End date (ISO format or epoch)
            sort: Sort order ('asc' or 'desc')
        
        Returns:
            List of RedditPost objects
        """
        endpoint = f"{self.BASE_URL}/api/posts/search"
        
        params = {
            'subreddit': subreddit,
            'limit': min(limit, 100),  # API max is 100
            'sort': sort
        }
        
        if after:
            params['after'] = after
        if before:
            params['before'] = before
        
        self._rate_limit()
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = [RedditPost.from_api_data(post) for post in data.get('data', [])]
            
            # Check rate limit headers
            remaining = response.headers.get('X-RateLimit-Remaining')
            if remaining:
                print(f"Rate limit remaining: {remaining}")
            
            return posts
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e.response, 'status_code'):
                print(f"Status code: {e.response.status_code}")
                if e.response.status_code == 429:
                    print("Rate limited - waiting 60 seconds")
                    time.sleep(60)
            return []
    
    def search_comments(self, subreddit: str, limit: int = 100,
                       after: Optional[str] = None,
                       before: Optional[str] = None,
                       sort: str = 'desc') -> List[RedditComment]:
        """
        Search for comments in a subreddit using Arctic Shift API
        
        Args:
            subreddit: Subreddit name (without r/)
            limit: Number of comments to retrieve (1-100)
            after: Start date (ISO format or epoch)
            before: End date (ISO format or epoch)
            sort: Sort order ('asc' or 'desc')
        
        Returns:
            List of RedditComment objects
        """
        endpoint = f"{self.BASE_URL}/api/comments/search"
        
        params = {
            'subreddit': subreddit,
            'limit': min(limit, 100),
            'sort': sort
        }
        
        if after:
            params['after'] = after
        if before:
            params['before'] = before
        
        self._rate_limit()
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            comments = [RedditComment.from_api_data(c) for c in data.get('data', [])]
            
            remaining = response.headers.get('X-RateLimit-Remaining')
            if remaining:
                print(f"Rate limit remaining: {remaining}")
            
            return comments
            
        except requests.exceptions.RequestException as e:
            print(f"API comment request failed: {e}")
            if hasattr(e.response, 'status_code'):
                print(f"Status code: {e.response.status_code}")
                if e.response.status_code == 429:
                    print("Rate limited - waiting 60 seconds")
                    time.sleep(60)
            return []
    
    def get_subreddit_stats(self, subreddit: str) -> Dict[str, Any]:
        """Get basic statistics for a subreddit"""
        endpoint = f"{self.BASE_URL}/api/time_series"
        
        params = {
            'key': f'r/{subreddit}/posts/count',
            'precision': 'month',
            'limit': 12  # Last 12 months
        }
        
        self._rate_limit()
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}


def save_to_json(posts: List[RedditPost], filename: str) -> None:
    """Save posts to JSON file"""
    data = [asdict(post) for post in posts]
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(posts)} posts to {filename}")


def save_to_csv(posts: List[RedditPost], filename: str) -> None:
    """Save posts to CSV file"""
    if not posts:
        print("No posts to save")
        return
    
    fieldnames = list(asdict(posts[0]).keys())
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for post in posts:
            writer.writerow(asdict(post))
    
    print(f"Saved {len(posts)} posts to {filename}")


def format_timestamp(timestamp: int) -> str:
    """Convert Unix timestamp to readable date"""
    if timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def display_posts(posts: List[RedditPost], limit: int = 10) -> None:
    """Display posts in console"""
    if not posts:
        print("No posts found")
        return
    
    print(f"\nFound {len(posts)} posts:")
    print("-" * 80)
    
    for i, post in enumerate(posts[:limit]):
        print(f"{i+1}. {post.title[:80]}...")
        print(f"   Author: u/{post.author} | Score: {post.score} | Comments: {post.num_comments}")
        print(f"   Date: {format_timestamp(post.created_utc)}")
        print(f"   URL: {post.url}")
        print()


def main():
    """Main function for command-line usage"""
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='Scrape Reddit posts from subreddit using Arctic Shift API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ethz --limit 50 --output posts.json
  %(prog)s stanford --after 2024-01-01 --format csv
  %(prog)s mit --stats --limit 20
        """
    )
    
    parser.add_argument('subreddit', help='Subreddit name (without r/)')
    parser.add_argument('--limit', type=int, default=50, 
                       help='Number of posts to retrieve (default: 50, use --max-posts for >100)')
    parser.add_argument('--max-posts', type=int, default=None,
                       help='Maximum posts to retrieve with pagination (overrides --limit)')
    parser.add_argument('--after', help='Start date (YYYY-MM-DD or epoch)')
    parser.add_argument('--before', help='End date (YYYY-MM-DD or epoch)')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--stats', action='store_true',
                       help='Show subreddit statistics')
    parser.add_argument('--display', type=int, default=10,
                       help='Number of posts to display in console (default: 10)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.limit > 100 and not args.max_posts:
        print("Warning: API limit is 100 posts per request. Using 100.")
        args.limit = 100
    
    # Initialize scraper
    scraper = ArcticShiftScraper()
    
    # Get subreddit statistics if requested
    if args.stats:
        print(f"\nFetching statistics for r/{args.subreddit}...")
        stats = scraper.get_subreddit_stats(args.subreddit)
        if stats:
            print(f"Statistics: {json.dumps(stats, indent=2)}")
        else:
            print("Could not retrieve statistics")
    
    # Search for posts
    print(f"\nSearching for posts in r/{args.subreddit}...")
    
    if args.max_posts or (args.after and args.before):
        # Use pagination for large requests or date ranges
        all_posts = []
        before_param = args.before
        limit_per_request = 100
        max_requests = 50  # Safety limit to prevent infinite loops
        
        for request_num in range(max_requests):
            print(f"  Fetching batch {request_num + 1} (posts {len(all_posts)}-{len(all_posts) + limit_per_request})...")
            
            batch = scraper.search_posts(
                subreddit=args.subreddit,
                limit=limit_per_request,
                after=args.after,
                before=before_param
            )
            
            if not batch:
                print("No more posts found")
                break
                
            all_posts.extend(batch)
            
            # Set before parameter for next request (oldest post timestamp)
            oldest_post = min(batch, key=lambda p: p.created_utc)
            before_param = str(oldest_post.created_utc)
            
            # Check if we've reached the start of the date range
            if args.after:
                try:
                    after_timestamp = int(args.after) if args.after.isdigit() else int(datetime.fromisoformat(args.after).timestamp())
                    if oldest_post.created_utc <= after_timestamp:
                        print(f"Reached start of date range ({format_timestamp(after_timestamp)})")
                        break
                except (ValueError, AttributeError):
                    pass
            
            # Stop if we've reached max posts
            if args.max_posts and len(all_posts) >= args.max_posts:
                print(f"Reached maximum posts limit ({args.max_posts})")
                all_posts = all_posts[:args.max_posts]
                break
            
            # Stop if we're getting duplicate/old posts (batch smaller than requested)
            if len(batch) < limit_per_request:
                print(f"Received fewer posts than requested ({len(batch)} < {limit_per_request}), stopping")
                break
            
            # Rate limiting between batches
            time.sleep(1.5)
        
        posts = all_posts
    else:
        # Single request (up to 100 posts)
        posts = scraper.search_posts(
            subreddit=args.subreddit,
            limit=args.limit,
            after=args.after,
            before=args.before
        )
    
    if not posts:
        print("No posts found. The subreddit may not exist or have no recent activity.")
        return 1
    
    # Display posts
    display_posts(posts, args.display)
    
    # Save to file if output specified
    if args.output:
        base_name = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        
        if args.format in ['json', 'both']:
            json_file = f"{base_name}.json"
            save_to_json(posts, json_file)
        
        if args.format in ['csv', 'both']:
            csv_file = f"{base_name}.csv"
            save_to_csv(posts, csv_file)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Subreddit: r/{args.subreddit}")
    print(f"  Posts retrieved: {len(posts)}")
    if posts:
        dates = [format_timestamp(p.created_utc) for p in posts]
        print(f"  Date range: {dates[-1]} to {dates[0]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())