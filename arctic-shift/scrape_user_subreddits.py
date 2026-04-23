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
from reddit_scraper import ArcticShiftScraper, RedditPost


@dataclass
class UserSubredditActivity:
    """Data class representing a user's activity across subreddits"""
    username: str
    subreddits: Dict[str, int]  # subreddit -> post count
    total_posts: int = 0
    first_post_date: Optional[int] = None
    last_post_date: Optional[int] = None
    
    def add_post(self, post: RedditPost) -> None:
        """Update user's subreddit activity with a new post"""
        self.total_posts += 1
        subreddit = post.subreddit
        
        if subreddit not in self.subreddits:
            self.subreddits[subreddit] = 0
        self.subreddits[subreddit] += 1
        
        if self.first_post_date is None or post.created_utc < self.first_post_date:
            self.first_post_date = post.created_utc
        if self.last_post_date is None or post.created_utc > self.last_post_date:
            self.last_post_date = post.created_utc


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
    
    def analyze_user_subreddits(self, username: str, max_posts: Optional[int] = None,
                               after: Optional[str] = None,
                               before: Optional[str] = None) -> UserSubredditActivity:
        """Analyze what subreddits a user posts in within optional date range"""
        all_posts: List[RedditPost] = []
        before_param: Optional[str] = before
        limit_per_request = 100
        
        print(f"  Analyzing u/{username}...", end="", flush=True)
        
        while True:
            # Stop if we've reached max posts (if specified)
            if max_posts and len(all_posts) >= max_posts:
                break
            
            posts = self.get_user_posts(username, limit=limit_per_request, 
                                       after=after, before=before_param)
            
            if not posts:
                break
            
            all_posts.extend(posts)
            
            # Set before parameter for next request (oldest post timestamp)
            oldest_post = min(posts, key=lambda p: p.created_utc)
            before_param = str(oldest_post.created_utc)
            
            # Stop if we're getting duplicate/old posts (batch smaller than requested)
            if len(posts) < limit_per_request:
                break
            
            # Rate limiting between batches
            time.sleep(1.2)
        
        # Create activity summary
        activity = UserSubredditActivity(username=username, subreddits={})
        for post in all_posts:
            activity.add_post(post)
        
        print(f" found {len(all_posts)} posts in {len(activity.subreddits)} subreddits")
        return activity
    
    def analyze_users_from_subreddit(self, target_subreddit: str, 
                                   user_limit: Optional[int] = None,
                                   posts_per_user: int = 100,
                                   after: Optional[str] = None,
                                   before: Optional[str] = None) -> Dict[str, UserSubredditActivity]:
        """
        Analyze users from a target subreddit to see what other subreddits they post in
        
        Args:
            target_subreddit: Subreddit to get users from
            user_limit: Maximum number of users to analyze (None = all users)
            posts_per_user: Maximum posts to fetch per user
            after: Start date (YYYY-MM-DD or epoch) for filtering posts
            before: End date (YYYY-MM-DD or epoch) for filtering posts
        
        Returns:
            Dictionary of UserSubredditActivity objects keyed by username
        """
        print(f"Analyzing users from r/{target_subreddit}...")
        
        # First, get users from the target subreddit within date range
        print("Step 1: Getting users from target subreddit...")
        target_posts = self.search_posts(
            subreddit=target_subreddit,
            limit=500,
            sort='desc',
            after=after,
            before=before
        )
        
        # Extract unique users
        users_from_target: Set[str] = set()
        for post in target_posts:
            if post.author and post.author != '[deleted]':
                users_from_target.add(post.author)
        
        print(f"Found {len(users_from_target)} unique users in r/{target_subreddit}")
        
        # Determine how many users to analyze
        users_to_analyze = list(users_from_target)
        if user_limit:
            users_to_analyze = users_to_analyze[:user_limit]
            print(f"\nStep 2: Analyzing cross-subreddit activity for {len(users_to_analyze)} users...")
        else:
            print(f"\nStep 2: Analyzing cross-subreddit activity for all {len(users_to_analyze)} users...")
        
        user_activities: Dict[str, UserSubredditActivity] = {}
        analyzed_count = 0
        
        for username in users_to_analyze:
            activity = self.analyze_user_subreddits(username, max_posts=posts_per_user,
                                                   after=after, before=before)
            if activity.total_posts > 0:
                user_activities[username] = activity
                analyzed_count += 1
            
            # Rate limiting between users
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
            'total_posts': activity.total_posts,
            'first_post_date': activity.first_post_date,
            'last_post_date': activity.last_post_date,
            'first_post_date_str': format_timestamp(activity.first_post_date),
            'last_post_date_str': format_timestamp(activity.last_post_date),
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
    
    # Prepare data
    rows = []
    for activity in activities.values():
        # Create a row for each user
        row = {
            'username': activity.username,
            'total_posts': activity.total_posts,
            'subreddit_count': len(activity.subreddits),
            'first_post_date': format_timestamp(activity.first_post_date),
            'last_post_date': format_timestamp(activity.last_post_date),
            'subreddits': ', '.join([f"{sub}:{count}" for sub, count in activity.subreddits.items()])
        }
        rows.append(row)
    
    fieldnames = ['username', 'total_posts', 'subreddit_count', 'first_post_date', 'last_post_date', 'subreddits']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved analysis of {len(activities)} users to {filename}")


def display_analysis(activities: Dict[str, UserSubredditActivity], 
                    target_subreddit: str,
                    display_limit: int = 20,
                    after: Optional[str] = None,
                    before: Optional[str] = None) -> None:
    """Display analysis results in console"""
    if not activities:
        print("No analysis to display")
        return
    
    # Show date range info if provided
    if after or before:
        date_range_str = "Date range: "
        if after:
            date_range_str += f"after {after} "
        if before:
            date_range_str += f"before {before}"
        print(date_range_str)
    
    # Sort users by number of subreddits they post in (descending)
    sorted_users = sorted(activities.values(), 
                         key=lambda u: len(u.subreddits), 
                         reverse=True)
    
    print(f"\nTop {min(display_limit, len(sorted_users))} users by subreddit diversity:")
    print("=" * 100)
    print(f"{'Username':<25} {'Subreddits':<8} {'Total Posts':<12} {'First Post':<12} {'Last Post':<12}")
    print("=" * 100)
    
    for i, activity in enumerate(sorted_users[:display_limit]):
        print(f"{activity.username:<25} {len(activity.subreddits):<8} {activity.total_posts:<12} "
              f"{format_timestamp(activity.first_post_date):<12} {format_timestamp(activity.last_post_date):<12}")
    
    # Summary statistics
    print(f"\nSummary statistics for users from r/{target_subreddit}:")
    print(f"  Total users analyzed: {len(activities)}")
    print(f"  Total posts analyzed: {sum(a.total_posts for a in activities.values())}")
    print(f"  Average posts per user: {sum(a.total_posts for a in activities.values()) / len(activities):.2f}")
    print(f"  Average subreddits per user: {sum(len(a.subreddits) for a in activities.values()) / len(activities):.2f}")
    
    # Most diverse user
    most_diverse = max(activities.values(), key=lambda u: len(u.subreddits))
    print(f"  Most diverse user: u/{most_diverse.username} ({len(most_diverse.subreddits)} subreddits)")
    
    # Find common subreddits among users
    print(f"\nCommon subreddits among users (excluding r/{target_subreddit}):")
    subreddit_counts: Dict[str, int] = {}
    for activity in activities.values():
        for subreddit in activity.subreddits.keys():
            if subreddit != target_subreddit:
                if subreddit not in subreddit_counts:
                    subreddit_counts[subreddit] = 0
                subreddit_counts[subreddit] += 1
    
    # Sort subreddits by user count (descending)
    sorted_subs = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Subreddit':<30} {'User Count':<12} {'% of Users':<10}")
    print("-" * 52)
    for subreddit, count in sorted_subs[:15]:
        percentage = (count / len(activities)) * 100
        print(f"r/{subreddit:<28} {count:<12} {percentage:.1f}%")


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
        activity = analyzer.analyze_user_subreddits(username, max_posts=posts_per_user,
                                                   after=after, before=before)
        if activity.total_posts > 0:
            user_activities[username] = activity
            analyzed_count += 1
        
        # Rate limiting between users
        time.sleep(0.5)
    
    print(f"\nAnalyzed {analyzed_count} users")
    return user_activities


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Analyze what other subreddits users post in',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all users from a subreddit (all posts per user)
  %(prog)s --subreddit ethz --output analysis.json
  
  # Analyze limited users from a subreddit (all posts per user)
  %(prog)s --subreddit ethz --user-limit 100 --output analysis.json
  
  # Analyze all users from a CSV file (all posts per user)
  %(prog)s --input ethz_users_2024-2026.csv
  
  # Analyze users from a CSV file with post limit
  %(prog)s --input ethz_users_2024-2026.csv --posts-per-user 200
  
  # Analyze users from a JSON file with date range
  %(prog)s --input ethz_users.json --after 2024-01-01 --before 2025-01-01
  
  # Analyze specific users with post limit
  %(prog)s --users user1,user2,user3 --posts-per-user 50
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--subreddit', help='Target subreddit name (without r/)')
    input_group.add_argument('--input', help='Input file with users (CSV or JSON from scrape_users.py)')
    input_group.add_argument('--users', help='Comma-separated list of usernames')
    
    parser.add_argument('--user-limit', type=int, 
                       help='Maximum number of users to analyze (only with --subreddit, default: all users)')
    parser.add_argument('--posts-per-user', type=int, 
                       help='Maximum posts to fetch per user (default: all posts)')
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
            posts_per_user=args.posts_per_user,
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
            posts_per_user=args.posts_per_user,
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
            posts_per_user=args.posts_per_user,
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
    
    # Display analysis
    display_analysis(activities, target_name, args.display, args.after, args.before)
    
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