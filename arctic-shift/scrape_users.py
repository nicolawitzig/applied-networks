#!/usr/bin/env python3
"""
Reddit Subreddit User Scraper using Arctic Shift API
Extracts unique users from posts in a subreddit
"""

import requests
import json
import csv
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
import sys
from reddit_scraper import ArcticShiftScraper, RedditPost


@dataclass
class RedditUser:
    """Data class representing a Reddit user"""
    username: str
    post_count: int = 0
    first_post_date: Optional[int] = None
    last_post_date: Optional[int] = None
    total_score: int = 0
    total_comments: int = 0
    
    def add_post(self, post: RedditPost) -> None:
        """Update user stats with a new post"""
        self.post_count += 1
        self.total_score += post.score
        self.total_comments += post.num_comments
        
        if self.first_post_date is None or post.created_utc < self.first_post_date:
            self.first_post_date = post.created_utc
        if self.last_post_date is None or post.created_utc > self.last_post_date:
            self.last_post_date = post.created_utc


class UserScraper(ArcticShiftScraper):
    """Extended scraper for collecting user data"""
    
    def collect_users_from_posts(self, posts: List[RedditPost]) -> Dict[str, RedditUser]:
        """Extract unique users from posts and aggregate their activity"""
        users: Dict[str, RedditUser] = {}
        
        for post in posts:
            if not post.author or post.author == '[deleted]':
                continue
                
            if post.author not in users:
                users[post.author] = RedditUser(username=post.author)
            
            users[post.author].add_post(post)
        
        return users
    
    def scrape_all_users(self, subreddit: str, max_posts: int = 1000,
                        after: Optional[str] = None,
                        before: Optional[str] = None) -> Dict[str, RedditUser]:
        """
        Scrape multiple pages of posts to collect users
        
        Args:
            subreddit: Subreddit name (without r/)
            max_posts: Maximum number of posts to scrape (0 = no limit)
            after: Start date (ISO format or epoch)
            before: End date (ISO format or epoch)
        
        Returns:
            Dictionary of RedditUser objects keyed by username
        """
        from datetime import datetime
        
        all_posts: List[RedditPost] = []
        users: Dict[str, RedditUser] = {}
        before_param = before
        limit_per_request = 100  # API max per request
        max_requests = 100  # Safety limit
        
        print(f"Scraping users from r/{subreddit}...")
        
        for request_num in range(max_requests):
            print(f"  Fetching batch {request_num + 1} (posts {len(all_posts)}-{len(all_posts) + limit_per_request})...")
            
            posts = self.search_posts(
                subreddit=subreddit,
                limit=limit_per_request,
                after=after,
                before=before_param,
                sort='desc'
            )
            
            if not posts:
                print("No more posts found")
                break
            
            # Extract users from this batch
            batch_users = self.collect_users_from_posts(posts)
            for username, user in batch_users.items():
                if username in users:
                    # Merge user data
                    existing = users[username]
                    existing.post_count += user.post_count
                    existing.total_score += user.total_score
                    existing.total_comments += user.total_comments
                    if user.first_post_date and (existing.first_post_date is None or user.first_post_date < existing.first_post_date):
                        existing.first_post_date = user.first_post_date
                    if user.last_post_date and (existing.last_post_date is None or user.last_post_date > existing.last_post_date):
                        existing.last_post_date = user.last_post_date
                else:
                    users[username] = user
            
            all_posts.extend(posts)
            
            # Set before parameter for next request (oldest post timestamp)
            oldest_post = min(posts, key=lambda p: p.created_utc)
            before_param = str(oldest_post.created_utc)
            
            # Check if we've reached the start of the date range
            if after:
                try:
                    after_timestamp = int(after) if after.isdigit() else int(datetime.fromisoformat(after).timestamp())
                    if oldest_post.created_utc <= after_timestamp:
                        print(f"Reached start of date range ({self._format_timestamp(after_timestamp)})")
                        break
                except (ValueError, AttributeError):
                    pass
            
            # Stop if we've reached max posts (only if max_posts > 0)
            if max_posts and len(all_posts) >= max_posts:
                print(f"Reached maximum posts limit ({max_posts})")
                all_posts = all_posts[:max_posts]
                break
            
            # Stop if we're getting duplicate/old posts (batch smaller than requested)
            if len(posts) < limit_per_request:
                print(f"Received fewer posts than requested ({len(posts)} < {limit_per_request}), stopping")
                break
            
            # Rate limiting between batches
            time.sleep(1.5)
        
        print(f"\nScraped {len(all_posts)} posts, found {len(users)} unique users")
        return users
    
    def _format_timestamp(self, timestamp: int) -> str:
        """Helper to format timestamp for display"""
        from datetime import datetime
        if timestamp == 0:
            return "N/A"
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def format_timestamp(timestamp: Optional[int]) -> str:
    """Convert Unix timestamp to readable date"""
    if timestamp is None or timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')


def save_users_to_json(users: Dict[str, RedditUser], filename: str) -> None:
    """Save users to JSON file"""
    data = []
    for user in users.values():
        user_dict = asdict(user)
        user_dict['first_post_date_str'] = format_timestamp(user.first_post_date)
        user_dict['last_post_date_str'] = format_timestamp(user.last_post_date)
        data.append(user_dict)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(users)} users to {filename}")


def save_users_to_csv(users: Dict[str, RedditUser], filename: str) -> None:
    """Save users to CSV file"""
    if not users:
        print("No users to save")
        return
    
    # Prepare data with formatted dates
    rows = []
    for user in users.values():
        row = asdict(user)
        row['first_post_date_str'] = format_timestamp(user.first_post_date)
        row['last_post_date_str'] = format_timestamp(user.last_post_date)
        rows.append(row)
    
    fieldnames = list(rows[0].keys())
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved {len(users)} users to {filename}")


def display_users(users: Dict[str, RedditUser], limit: int = 20) -> None:
    """Display users in console"""
    if not users:
        print("No users found")
        return
    
    # Sort users by post count (descending)
    sorted_users = sorted(users.values(), key=lambda u: u.post_count, reverse=True)
    
    print(f"\nTop {min(limit, len(sorted_users))} users by post count:")
    print("-" * 100)
    print(f"{'Username':<25} {'Posts':<8} {'Total Score':<12} {'Total Comments':<15} {'First Post':<12} {'Last Post':<12}")
    print("-" * 100)
    
    for i, user in enumerate(sorted_users[:limit]):
        print(f"{user.username:<25} {user.post_count:<8} {user.total_score:<12} {user.total_comments:<15} "
              f"{format_timestamp(user.first_post_date):<12} {format_timestamp(user.last_post_date):<12}")
    
    # Summary statistics
    print(f"\nSummary statistics:")
    print(f"  Total unique users: {len(users)}")
    print(f"  Total posts analyzed: {sum(u.post_count for u in users.values())}")
    print(f"  Average posts per user: {sum(u.post_count for u in users.values()) / len(users):.2f}")
    
    # Most active user
    most_active = max(users.values(), key=lambda u: u.post_count)
    print(f"  Most active user: u/{most_active.username} ({most_active.post_count} posts)")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Scrape Reddit users from subreddit using Arctic Shift API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ethz --max-posts 500 --output users.json
  %(prog)s stanford --format csv --display 30
  %(prog)s mit --max-posts 1000 --format both
        """
    )
    
    parser.add_argument('subreddit', help='Subreddit name (without r/)')
    parser.add_argument('--max-posts', type=int, default=500,
                       help='Maximum number of posts to scrape (default: 500)')
    parser.add_argument('--after', help='Start date (YYYY-MM-DD or epoch)')
    parser.add_argument('--before', help='End date (YYYY-MM-DD or epoch)')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--display', type=int, default=20,
                       help='Number of users to display in console (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = UserScraper()
    
    # Scrape users
    # If date range is specified but no max_posts, get all posts in range
    effective_max_posts = args.max_posts
    if (args.after or args.before) and args.max_posts == 500:  # Default value
        print("Date range specified, getting all posts in range (override --max-posts to limit)")
        effective_max_posts = 0  # No limit
    
    users = scraper.scrape_all_users(
        subreddit=args.subreddit,
        max_posts=effective_max_posts,
        after=args.after,
        before=args.before
    )
    
    if not users:
        print("No users found. The subreddit may not exist or have no recent activity.")
        return 1
    
    # Display users
    display_users(users, args.display)
    
    # Save to file if output specified
    if args.output:
        base_name = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        
        if args.format in ['json', 'both']:
            json_file = f"{base_name}.json"
            save_users_to_json(users, json_file)
        
        if args.format in ['csv', 'both']:
            csv_file = f"{base_name}.csv"
            save_users_to_csv(users, csv_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())