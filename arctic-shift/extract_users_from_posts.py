#!/usr/bin/env python3
"""
Extract unique users from existing Reddit post files (CSV/JSON)
Useful when you already have posts data and want to analyze users
"""

import json
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import sys


@dataclass
class RedditUser:
    """Data class representing a Reddit user extracted from posts"""
    username: str
    post_count: int = 0
    first_post_date: Optional[int] = None
    last_post_date: Optional[int] = None
    total_score: int = 0
    total_comments: int = 0
    
    def add_post(self, post_data: Dict) -> None:
        """Update user stats with a new post"""
        self.post_count += 1
        self.total_score += post_data.get('score', 0)
        self.total_comments += post_data.get('num_comments', 0)
        
        created_utc = post_data.get('created_utc', 0)
        if self.first_post_date is None or created_utc < self.first_post_date:
            self.first_post_date = created_utc
        if self.last_post_date is None or created_utc > self.last_post_date:
            self.last_post_date = created_utc


def load_posts_from_json(filename: str) -> List[Dict]:
    """Load posts from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_posts_from_csv(filename: str) -> List[Dict]:
    """Load posts from CSV file"""
    posts = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string fields to appropriate types
            row['created_utc'] = int(row['created_utc']) if row.get('created_utc') else 0
            row['score'] = int(row['score']) if row.get('score') else 0
            row['num_comments'] = int(row['num_comments']) if row.get('num_comments') else 0
            row['retrieved_on'] = int(row['retrieved_on']) if row.get('retrieved_on') else 0
            posts.append(row)
    return posts


def extract_users_from_posts(posts: List[Dict]) -> Dict[str, RedditUser]:
    """Extract unique users from posts"""
    users: Dict[str, RedditUser] = {}
    
    for post in posts:
        author = post.get('author', '')
        if not author or author == '[deleted]':
            continue
            
        if author not in users:
            users[author] = RedditUser(username=author)
        
        users[author].add_post(post)
    
    return users


def filter_users_by_date(users: Dict[str, RedditUser], 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Dict[str, RedditUser]:
    """Filter users based on their activity dates"""
    if not start_date and not end_date:
        return users
    
    filtered_users = {}
    
    # Convert date strings to timestamps
    start_timestamp = None
    end_timestamp = None
    
    if start_date:
        try:
            start_timestamp = int(start_date) if start_date.isdigit() else int(datetime.fromisoformat(start_date).timestamp())
        except (ValueError, AttributeError):
            print(f"Warning: Invalid start date format: {start_date}")
    
    if end_date:
        try:
            end_timestamp = int(end_date) if end_date.isdigit() else int(datetime.fromisoformat(end_date).timestamp())
        except (ValueError, AttributeError):
            print(f"Warning: Invalid end date format: {end_date}")
    
    for username, user in users.items():
        # Check if user has posts within date range
        if start_timestamp and user.last_post_date and user.last_post_date < start_timestamp:
            continue  # User's last post is before start date
        if end_timestamp and user.first_post_date and user.first_post_date > end_timestamp:
            continue  # User's first post is after end date
        
        filtered_users[username] = user
    
    return filtered_users


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
    if users:
        most_active = max(users.values(), key=lambda u: u.post_count)
        print(f"  Most active user: u/{most_active.username} ({most_active.post_count} posts)")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Extract unique users from existing Reddit post files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ethz_posts.json --output ethz_users.json
  %(prog)s ethz_posts.csv --format csv --display 30
  %(prog)s posts.json --after 2024-01-01 --before 2025-01-01
        """
    )
    
    parser.add_argument('input_file', help='Input file with posts (JSON or CSV)')
    parser.add_argument('--output', help='Output filename for users')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--display', type=int, default=20,
                       help='Number of users to display in console (default: 20)')
    parser.add_argument('--after', help='Filter users active after this date (YYYY-MM-DD or epoch)')
    parser.add_argument('--before', help='Filter users active before this date (YYYY-MM-DD or epoch)')
    
    args = parser.parse_args()
    
    # Load posts from file
    print(f"Loading posts from {args.input_file}...")
    
    if args.input_file.endswith('.json'):
        posts = load_posts_from_json(args.input_file)
    elif args.input_file.endswith('.csv'):
        posts = load_posts_from_csv(args.input_file)
    else:
        print(f"Error: Unsupported file format. Use .json or .csv")
        return 1
    
    print(f"Loaded {len(posts)} posts")
    
    # Extract users
    users = extract_users_from_posts(posts)
    print(f"Found {len(users)} unique users")
    
    # Filter by date if specified
    if args.after or args.before:
        users = filter_users_by_date(users, args.after, args.before)
        print(f"After date filtering: {len(users)} users")
    
    if not users:
        print("No users found after filtering")
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