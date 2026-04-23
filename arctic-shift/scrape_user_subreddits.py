#!/usr/bin/env python3
"""
Reddit User Cross-Subreddit Analysis
For each user from a target subreddit, find what other subreddits they post in
"""

import requests
import json
import csv
import time
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from reddit_scraper import ArcticShiftScraper, RedditPost, RedditComment


@dataclass
class UserSubredditActivity:
    """Data class representing a user's activity across subreddits (posts + comments)"""
    username: str
    subreddits: Dict[str, int]  # subreddit -> combined post+comment count
    total_items: int = 0
    total_posts: int = 0
    total_comments: int = 0
    first_date: Optional[int] = None
    last_date: Optional[int] = None
    
    def add_post(self, post: RedditPost) -> None:
        """Update user's subreddit activity with a new post"""
        self.total_items += 1
        self.total_posts += 1
        subreddit = post.subreddit
        
        if subreddit not in self.subreddits:
            self.subreddits[subreddit] = 0
        self.subreddits[subreddit] += 1
        
        if self.first_date is None or post.created_utc < self.first_date:
            self.first_date = post.created_utc
        if self.last_date is None or post.created_utc > self.last_date:
            self.last_date = post.created_utc
    
    def add_comment(self, comment: RedditComment) -> None:
        """Update user's subreddit activity with a new comment"""
        self.total_items += 1
        self.total_comments += 1
        subreddit = comment.subreddit
        
        if subreddit not in self.subreddits:
            self.subreddits[subreddit] = 0
        self.subreddits[subreddit] += 1
        
        if self.first_date is None or comment.created_utc < self.first_date:
            self.first_date = comment.created_utc
        if self.last_date is None or comment.created_utc > self.last_date:
            self.last_date = comment.created_utc


class CrossSubredditAnalyzer(ArcticShiftScraper):
    """Analyzer for finding what subreddits users post in"""
    
    def get_user_posts(self, username: str, limit: int = 100,
                      after: Optional[str] = None,
                      before: Optional[str] = None) -> List[RedditPost]:
        """Get posts by a specific user within optional date range"""
        endpoint = f"{self.BASE_URL}/api/posts/search"
        
        params = {
            'author': username,
            'limit': min(limit, 100),
            'sort': 'desc'
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
            
            return posts
            
        except requests.exceptions.RequestException as e:
            print(f"API request for user {username} failed: {e}")
            if hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    print("Rate limited - waiting 60 seconds")
                    time.sleep(60)
            return []
    
    def get_user_comments(self, username: str, limit: int = 100,
                         after: Optional[str] = None,
                         before: Optional[str] = None) -> List[RedditComment]:
        """Get comments by a specific user within optional date range"""
        endpoint = f"{self.BASE_URL}/api/comments/search"
        
        params = {
            'author': username,
            'limit': min(limit, 100),
            'sort': 'desc'
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
            
            return comments
            
        except requests.exceptions.RequestException as e:
            print(f"API request for user {username} comments failed: {e}")
            if hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    print("Rate limited - waiting 60 seconds")
                    time.sleep(60)
            return []
    
    def _paginate_items(self, fetch_fn, username: str, max_items: Optional[int],
                        after: Optional[str], before: Optional[str],
                        item_type: str) -> tuple:
        """
        Generic pagination helper that fetches items for a user until exhausted or max_items reached.
        
        Args:
            fetch_fn: Function to call for each batch (get_user_posts or get_user_comments)
            username: Username to fetch for
            max_items: Maximum number of items to fetch (None = no limit)
            after: Start date filter
            before: End date filter
            item_type: 'posts' or 'comments' (for display)
        
        Returns:
            Tuple of (all_items, count) where count is total fetched
        """
        all_items: list = []
        before_param: Optional[str] = before
        limit_per_request = 100
        
        while True:
            if max_items and len(all_items) >= max_items:
                break
            
            items = fetch_fn(username, limit=limit_per_request,
                            after=after, before=before_param)
            
            if not items:
                break
            
            all_items.extend(items)
            
            oldest = min(items, key=lambda p: p.created_utc)
            before_param = str(oldest.created_utc)
            
            if len(items) < limit_per_request:
                break
            
            time.sleep(1.2)
        
        return all_items, len(all_items)
    
    def analyze_user_subreddits(self, username: str, max_items: Optional[int] = None,
                               after: Optional[str] = None,
                               before: Optional[str] = None) -> UserSubredditActivity:
        """Analyze what subreddits a user posts and comments in within optional date range"""
        print(f"  Analyzing u/{username}...", end="", flush=True)
        
        # Split max_items roughly evenly between posts and comments
        if max_items:
            half = max_items // 2
            posts, _ = self._paginate_items(self.get_user_posts, username, half,
                                           after, before, 'posts')
            remaining = max_items - len(posts)
            comments, _ = self._paginate_items(self.get_user_comments, username, remaining,
                                              after, before, 'comments')
        else:
            posts, _ = self._paginate_items(self.get_user_posts, username, None,
                                           after, before, 'posts')
            comments, _ = self._paginate_items(self.get_user_comments, username, None,
                                              after, before, 'comments')
        
        activity = UserSubredditActivity(username=username, subreddits={})
        for post in posts:
            activity.add_post(post)
        for comment in comments:
            activity.add_comment(comment)
        
        print(f" found {len(posts)} posts + {len(comments)} comments = {activity.total_items} items in {len(activity.subreddits)} subreddits")
        return activity
    
    def analyze_users_from_subreddit(self, target_subreddit: str, 
                                    user_limit: Optional[int] = None,
                                    posts_per_user: int = 100,
                                    after: Optional[str] = None,
                                    before: Optional[str] = None) -> Dict[str, UserSubredditActivity]:
        """
        Analyze users from a target subreddit to see what other subreddits they are active in
        
        Discovers users who have posted OR commented in the target subreddit,
        then fetches both their posts and comments across all subreddits.
        
        Args:
            target_subreddit: Subreddit to get users from
            user_limit: Maximum number of users to analyze (None = all users)
            posts_per_user: Maximum items (posts + comments combined) to fetch per user
            after: Start date (YYYY-MM-DD or epoch) for filtering
            before: End date (YYYY-MM-DD or epoch) for filtering
        
        Returns:
            Dictionary of UserSubredditActivity objects keyed by username
        """
        print(f"Analyzing users from r/{target_subreddit}...")
        
        print("Step 1: Getting users from posts in target subreddit...")
        target_posts = self.search_posts(
            subreddit=target_subreddit,
            limit=500,
            sort='desc',
            after=after,
            before=before
        )
        
        users_from_target: Set[str] = set()
        for post in target_posts:
            if post.author and post.author != '[deleted]':
                users_from_target.add(post.author)
        
        print(f"  Found {len(users_from_target)} unique users from posts")
        
        print("Step 1b: Getting users from comments in target subreddit...")
        target_comments = self.search_comments(
            subreddit=target_subreddit,
            limit=500,
            sort='desc',
            after=after,
            before=before
        )
        
        comment_authors: Set[str] = set()
        for comment in target_comments:
            if comment.author and comment.author != '[deleted]':
                comment_authors.add(comment.author)
        
        new_from_comments = comment_authors - users_from_target
        users_from_target.update(comment_authors)
        
        print(f"  Found {len(comment_authors)} unique users from comments ({len(new_from_comments)} new, not in posts)")
        print(f"Total: {len(users_from_target)} unique users in r/{target_subreddit}")
        
        users_to_analyze = list(users_from_target)
        if user_limit:
            users_to_analyze = users_to_analyze[:user_limit]
            print(f"\nStep 2: Analyzing cross-subreddit activity for {len(users_to_analyze)} users...")
        else:
            print(f"\nStep 2: Analyzing cross-subreddit activity for all {len(users_to_analyze)} users...")
        
        user_activities: Dict[str, UserSubredditActivity] = {}
        analyzed_count = 0
        
        for username in users_to_analyze:
            activity = self.analyze_user_subreddits(username, max_items=posts_per_user,
                                                   after=after, before=before)
            if activity.total_items > 0:
                user_activities[username] = activity
                analyzed_count += 1
            
            time.sleep(0.5)
        
        print(f"\nAnalyzed {analyzed_count} users")
        return user_activities


def format_timestamp(timestamp: Optional[int]) -> str:
    """Convert Unix timestamp to readable date"""
    if timestamp is None or timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')


def save_analysis_to_json(activities: Dict[str, UserSubredditActivity], filename: str) -> None:
    """Save analysis to JSON file"""
    data = []
    for activity in activities.values():
        activity_dict = {
            'username': activity.username,
            'total_items': activity.total_items,
            'total_posts': activity.total_posts,
            'total_comments': activity.total_comments,
            'first_date': activity.first_date,
            'last_date': activity.last_date,
            'first_date_str': format_timestamp(activity.first_date),
            'last_date_str': format_timestamp(activity.last_date),
            'subreddits': activity.subreddits,
            'subreddit_count': len(activity.subreddits)
        }
        data.append(activity_dict)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved analysis of {len(activities)} users to {filename}")


def save_analysis_to_csv(activities: Dict[str, UserSubredditActivity], filename: str) -> None:
    """Save analysis to CSV file"""
    if not activities:
        print("No analysis to save")
        return
    
    rows = []
    for activity in activities.values():
        row = {
            'username': activity.username,
            'total_items': activity.total_items,
            'total_posts': activity.total_posts,
            'total_comments': activity.total_comments,
            'subreddit_count': len(activity.subreddits),
            'first_date': format_timestamp(activity.first_date),
            'last_date': format_timestamp(activity.last_date),
            'subreddits': ', '.join([f"{sub}:{count}" for sub, count in activity.subreddits.items()])
        }
        rows.append(row)
    
    fieldnames = ['username', 'total_items', 'total_posts', 'total_comments', 'subreddit_count', 'first_date', 'last_date', 'subreddits']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved analysis of {len(activities)} users to {filename}")



def load_users_from_csv(filename: str) -> List[str]:
    """Load usernames from CSV file (output of scrape_users.py)"""
    usernames = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'username' in row and row['username']:
                    usernames.append(row['username'])
        print(f"Loaded {len(usernames)} users from {filename}")
        return usernames
    except Exception as e:
        print(f"Error loading users from CSV: {e}")
        return []


def load_users_from_json(filename: str) -> List[str]:
    """Load usernames from JSON file (output of scrape_users.py)"""
    usernames = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict) and 'username' in item:
                    usernames.append(item['username'])
        print(f"Loaded {len(usernames)} users from {filename}")
        return usernames
    except Exception as e:
        print(f"Error loading users from JSON: {e}")
        return []


def analyze_users_from_file(analyzer: CrossSubredditAnalyzer,
                          usernames: List[str],
                          posts_per_user: Optional[int] = None,
                          after: Optional[str] = None,
                          before: Optional[str] = None) -> Dict[str, UserSubredditActivity]:
    """Analyze cross-subreddit activity for users from a file"""
    print(f"Analyzing {len(usernames)} users...")
    
    user_activities: Dict[str, UserSubredditActivity] = {}
    analyzed_count = 0
    
    for username in usernames:
        activity = analyzer.analyze_user_subreddits(username, max_items=posts_per_user,
                                                   after=after, before=before)
        if activity.total_items > 0:
            user_activities[username] = activity
            analyzed_count += 1
        
        time.sleep(0.5)
    
    print(f"\nAnalyzed {analyzed_count} users")
    return user_activities


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Analyze what other subreddits users post and comment in',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all users from a subreddit (discovers from both posts and comments)
  %(prog)s --subreddit ethz --output analysis.json
  
  # Analyze limited users from a subreddit
  %(prog)s --subreddit ethz --user-limit 100 --output analysis.json
  
  # Analyze users from a CSV file (limit combined posts+comments per user)
  %(prog)s --input ethz_users_2024-2026.csv --posts-per-user 200
  
  # Analyze users from a JSON file with date range
  %(prog)s --input ethz_users.json --after 2024-01-01 --before 2025-01-01
  
  # Analyze specific users with item limit (50 posts + 50 comments = 100 total)
  %(prog)s --users user1,user2,user3 --posts-per-user 100
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--subreddit', help='Target subreddit name (without r/)')
    input_group.add_argument('--input', help='Input file with users (CSV or JSON from scrape_users.py)')
    input_group.add_argument('--users', help='Comma-separated list of usernames')
    
    parser.add_argument('--user-limit', type=int, 
                       help='Maximum number of users to analyze (only with --subreddit, default: all users)')
    parser.add_argument('--posts-per-user', type=int, dest='max_items',
                       help='Maximum items (posts + comments combined) to fetch per user (default: all)')
    parser.add_argument('--after', help='Start date (YYYY-MM-DD or epoch) for filtering posts')
    parser.add_argument('--before', help='End date (YYYY-MM-DD or epoch) for filtering posts')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--display', type=int, default=20,
                       help='Number of users to display in console (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CrossSubredditAnalyzer()
    
    # Get users based on input method
    if args.subreddit:
        # Analyze users from subreddit
        activities = analyzer.analyze_users_from_subreddit(
            target_subreddit=args.subreddit,
            user_limit=args.user_limit,
            posts_per_user=args.max_items,
            after=args.after,
            before=args.before
        )
        target_name = args.subreddit
    elif args.input:
        # Load users from file
        if args.input.lower().endswith('.csv'):
            usernames = load_users_from_csv(args.input)
        elif args.input.lower().endswith('.json'):
            usernames = load_users_from_json(args.input)
        else:
            print(f"Unsupported file format: {args.input}. Use .csv or .json")
            return 1
        
        if not usernames:
            print("No users loaded from file")
            return 1
        
        activities = analyze_users_from_file(
            analyzer=analyzer,
            usernames=usernames,
            posts_per_user=args.max_items,
            after=args.after,
            before=args.before
        )
        target_name = os.path.basename(args.input).rsplit('.', 1)[0]
    elif args.users:
        # Use comma-separated list of users
        usernames = [u.strip() for u in args.users.split(',') if u.strip()]
        print(f"Analyzing {len(usernames)} specified users")
        
        activities = analyze_users_from_file(
            analyzer=analyzer,
            usernames=usernames,
            posts_per_user=args.max_items,
            after=args.after,
            before=args.before
        )
        target_name = f"{len(usernames)}_users"
    else:
        print("No input method specified")
        return 1
    
    if not activities:
        print("No user activity found.")
        return 1

    # Save to file if output specified
    if args.output:
        base_name = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        
        if args.format in ['json', 'both']:
            json_file = f"{base_name}.json"
            save_analysis_to_json(activities, json_file)
        
        if args.format in ['csv', 'both']:
            csv_file = f"{base_name}.csv"
            save_analysis_to_csv(activities, csv_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())