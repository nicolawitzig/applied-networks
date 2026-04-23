#!/usr/bin/env python3
"""
Test script for Reddit scraper
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reddit_scraper import ArcticShiftScraper, RedditPost, display_posts


def test_basic_functionality():
    """Test basic scraper functionality"""
    print("Testing Arctic Shift API scraper...")
    print("=" * 60)
    
    # Initialize scraper
    scraper = ArcticShiftScraper()
    
    # Test with a popular university subreddit
    test_subreddit = "berkeley"  # r/berkeley
    
    print(f"\n1. Testing post search for r/{test_subreddit}...")
    posts = scraper.search_posts(
        subreddit=test_subreddit,
        limit=10,
        sort='desc'
    )
    
    if posts:
        print(f"✓ Successfully retrieved {len(posts)} posts")
        display_posts(posts, limit=3)
        
        # Show sample data
        print("\n2. Sample post data:")
        if posts:
            post = posts[0]
            print(f"   Title: {post.title[:100]}...")
            print(f"   Author: u/{post.author}")
            print(f"   Score: {post.score}")
            print(f"   Comments: {post.num_comments}")
            print(f"   Date: {post.created_utc}")
            
        return True
    else:
        print("✗ No posts retrieved. Possible issues:")
        print("  - Subreddit may not exist in dataset")
        print("  - API may be temporarily unavailable")
        print("  - Network connection issue")
        return False


def test_multiple_subreddits():
    """Test scraping multiple university subreddits"""
    print("\n" + "=" * 60)
    print("Testing multiple university subreddits...")
    
    scraper = ArcticShiftScraper()
    universities = ["stanford", "mit", "harvard"]
    
    for uni in universities:
        print(f"\nTesting r/{uni}...")
        posts = scraper.search_posts(subreddit=uni, limit=5)
        
        if posts:
            print(f"  ✓ Found {len(posts)} posts")
            # Show most recent post
            if posts:
                latest = posts[0]
                print(f"  Latest: '{latest.title[:60]}...' by u/{latest.author}")
        else:
            print(f"  ✗ No posts found")
        
        # Be respectful with rate limiting
        import time
        time.sleep(1)


def test_date_filtering():
    """Test date-based filtering"""
    print("\n" + "=" * 60)
    print("Testing date filtering...")
    
    scraper = ArcticShiftScraper()
    
    # Get posts from 2024 only
    posts = scraper.search_posts(
        subreddit="berkeley",
        limit=5,
        after="2024-01-01",
        before="2025-01-01"
    )
    
    if posts:
        print(f"Found {len(posts)} posts from 2024")
        for post in posts:
            from datetime import datetime
            date_str = datetime.fromtimestamp(post.created_utc).strftime('%Y-%m')
            print(f"  - {date_str}: {post.title[:50]}...")


if __name__ == "__main__":
    print("Arctic Shift API Scraper Test")
    print("=" * 60)
    
    # Run tests
    test_basic_functionality()
    test_multiple_subreddits()
    test_date_filtering()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nUsage examples:")
    print("  python reddit_scraper.py ethz --limit 20 --output ethz_posts.json")
    print("  python reddit_scraper.py stanford --after 2024-01-01 --format csv")
    print("  python reddit_scraper.py mit --stats --display 5")