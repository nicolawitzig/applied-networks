# Reddit Scraper Guide for Arctic Shift API

This guide explains how to use the Reddit scraping scripts to analyze subreddit users and their cross-subreddit activity.

## Scripts Overview

1. **`reddit_scraper.py`** - Scrape posts from a subreddit with pagination and date ranges
2. **`scrape_users.py`** - Extract unique users from a subreddit (with date range support)
3. **`extract_users_from_posts.py`** - Extract users from existing post files (CSV/JSON)
4. **`scrape_user_subreddits.py`** - Analyze what other subreddits users post in
5. **`analyze_overlap.py`** - Analyze subreddit overlap patterns from collected data

## Prerequisites

```bash
# Install required packages
pip install requests
```

## Basic Usage

### 1. Scrape Posts from a Subreddit

```bash
# Get 50 posts from r/ethz (default)
python reddit_scraper.py ethz

# Get 100 posts and save to JSON
python reddit_scraper.py ethz --limit 100 --output ethz_posts.json

# Get posts from a date range
python reddit_scraper.py ethz --after 2024-01-01 --before 2024-12-31

# Get more than 100 posts using pagination
python reddit_scraper.py ethz --max-posts 500 --output ethz_posts_large.json

# Get 1000 posts with date range
python reddit_scraper.py ethz --max-posts 1000 --after 2024-01-01 --before 2026-01-01

# Save in multiple formats
python reddit_scraper.py ethz --output data --format both
```

### 2. Extract Unique Users from a Subreddit

```bash
# Get users from r/ethz (scrapes up to 1000 posts)
python scrape_users.py ethz --max-posts 1000

# Get users with date range
python scrape_users.py ethz --after 2024-01-01 --before 2026-01-01 --max-posts 2000

# Save users to files
python scrape_users.py ethz --max-posts 1000 --output ethz_users --format both

# Display top 30 users
python scrape_users.py ethz --max-posts 500 --display 30
```

### 2b. Extract Users from Existing Post Files

```bash
# Extract users from JSON post file
python extract_users_from_posts.py ethz_posts.json --output ethz_users.json

# Extract users from CSV post file
python extract_users_from_posts.py ethz_posts.csv --format csv --display 30

# Filter users by date range
python extract_users_from_posts.py ethz_posts.json --after 2024-01-01 --before 2025-01-01 --output filtered_users.json
```

### 3. Analyze Cross-Subreddit Activity

```bash
# Analyze all users from r/ethz
python scrape_user_subreddits.py --subreddit ethz --posts-per-user 100

# Analyze limited users from r/ethz
python scrape_user_subreddits.py --subreddit ethz --user-limit 50 --posts-per-user 100

# Save comprehensive analysis
python scrape_user_subreddits.py --subreddit ethz --user-limit 100 --posts-per-user 100 --output ethz_cross_analysis --format both

# Analyze users within a specific date range
python scrape_user_subreddits.py --subreddit ethz --after 2024-01-01 --before 2025-01-01 --user-limit 100

# Analyze recent activity only
python scrape_user_subreddits.py --subreddit ethz --after 2024-06-01 --before 2024-12-31 --posts-per-user 200

# Analyze all users from a CSV file (output of scrape_users.py)
python scrape_user_subreddits.py --input ethz_users_2024-2026.csv --posts-per-user 200

# Analyze users from a JSON file with date range
python scrape_user_subreddits.py --input ethz_users.json --after 2024-01-01 --before 2025-01-01

# Analyze specific users
python scrape_user_subreddits.py --users user1,user2,user3 --posts-per-user 50
```

### 4. Analyze Subreddit Overlap

```bash
# Analyze overlap from saved analysis
python analyze_overlap.py ethz_cross_subreddit_analysis.json ethz_overlap
```

## Output Files

### From `scrape_users.py` or `extract_users_from_posts.py`:
- `ethz_users.json` - JSON with user statistics (post count, activity dates, etc.)
- `ethz_users.csv` - CSV version of user data

### From `scrape_user_subreddits.py`:
- `ethz_cross_subreddit_analysis.json` - Detailed JSON with user activity across subreddits
- `ethz_cross_subreddit_analysis.csv` - CSV summary of user-subreddit relationships

### From `analyze_overlap.py`:
- `ethz_overlap_subreddit_overlap.csv` - Subreddit overlap counts
- `ethz_overlap_pairwise_overlap.csv` - Pairwise overlap between subreddits

## Command Line Arguments

### `reddit_scraper.py`
```
positional arguments:
  subreddit            Subreddit name (without r/)

optional arguments:
  --limit LIMIT        Number of posts to retrieve (default: 50, max 100 per request)
  --max-posts MAX_POSTS Maximum posts to retrieve with pagination (overrides --limit)
  --after AFTER        Start date (YYYY-MM-DD or epoch)
  --before BEFORE      End date (YYYY-MM-DD or epoch)
  --output OUTPUT      Output filename
  --format {json,csv,both} Output format (default: json)
  --stats              Show subreddit statistics
  --display DISPLAY    Number of posts to display in console (default: 10)
```

### `scrape_users.py`
```
positional arguments:
  subreddit            Subreddit name (without r/)

optional arguments:
  --max-posts MAX_POSTS Maximum number of posts to scrape (default: 500)
  --after AFTER        Start date (YYYY-MM-DD or epoch)
  --before BEFORE      End date (YYYY-MM-DD or epoch)
  --output OUTPUT      Output filename
  --format {json,csv,both} Output format (default: json)
  --display DISPLAY    Number of users to display in console (default: 20)
```

### `extract_users_from_posts.py`
```
positional arguments:
  input_file           Input file with posts (JSON or CSV)

optional arguments:
  --output OUTPUT      Output filename for users
  --format {json,csv,both} Output format (default: json)
  --display DISPLAY    Number of users to display in console (default: 20)
  --after AFTER        Filter users active after this date (YYYY-MM-DD or epoch)
  --before BEFORE      Filter users active before this date (YYYY-MM-DD or epoch)
```

### `scrape_user_subreddits.py`
```
options:
  --subreddit SUBREDDIT  Target subreddit name (without r/)
  --input INPUT         Input file with users (CSV or JSON from scrape_users.py)
  --users USERS         Comma-separated list of usernames
  --user-limit USER_LIMIT Maximum number of users to analyze (only with --subreddit, default: all users)
  --posts-per-user POSTS_PER_USER Maximum posts to fetch per user (default: 100)
  --after AFTER        Start date (YYYY-MM-DD or epoch) for filtering posts
  --before BEFORE      End date (YYYY-MM-DD or epoch) for filtering posts
  --output OUTPUT      Output filename
  --format {json,csv,both} Output format (default: json)
  --display DISPLAY    Number of users to display in console (default: 20)
```

### `analyze_overlap.py`
```
usage: python analyze_overlap.py <analysis_file.json> [output_prefix]

Example: python analyze_overlap.py ethz_cross_subreddit_analysis.json ethz_overlap
```

## Example Workflow

### Complete Analysis of r/ethz Users

```bash
# Step 1: Extract users from r/ethz (with pagination)
python scrape_users.py ethz --max-posts 1000 --output ethz_users --format both

# Alternative: Use updated reddit_scraper.py with pagination
python reddit_scraper.py ethz --max-posts 1000 --output ethz_posts_large.json

# Step 2: Analyze cross-subreddit activity (from subreddit, all users)
python scrape_user_subreddits.py --subreddit ethz --posts-per-user 100 --output ethz_cross_analysis --format both

# Alternative: Analyze limited users from subreddit
python scrape_user_subreddits.py --subreddit ethz --user-limit 100 --posts-per-user 100 --output ethz_cross_analysis --format both

# Alternative: Analyze all users from saved file
python scrape_user_subreddits.py --input ethz_users.csv --posts-per-user 200 --output ethz_cross_analysis

# Step 3: Analyze overlap patterns
python analyze_overlap.py ethz_cross_subreddit_analysis.json ethz_overlap

# Alternative: Analyze specific time period
python scrape_user_subreddits.py --subreddit ethz --after 2024-01-01 --before 2025-01-01 --user-limit 100 --posts-per-user 200 --output ethz_2024_analysis
```

### Large-Scale Data Collection

```bash
# Collect 5000 posts for comprehensive analysis
python reddit_scraper.py ethz --max-posts 5000 --output ethz_5k_posts.json

# Analyze users from large dataset
python scrape_users.py ethz --max-posts 5000 --output ethz_users_large --format both
```

### Quick Analysis

```bash
# Quick analysis of top 30 users from subreddit
python scrape_user_subreddits.py --subreddit ethz --user-limit 30 --posts-per-user 50 --display 20

# Quick analysis of users from file
python scrape_user_subreddits.py --input ethz_users.csv --posts-per-user 50 --display 20
```

## Data Structure

### User Data (`scrape_users.py` output)
```json
{
  "username": "Visible-Design4180",
  "post_count": 12,
  "first_post_date": 1776563865,
  "last_post_date": 1776693493,
  "total_score": 47,
  "total_comments": 70
}
```

### Cross-Subreddit Analysis (`scrape_user_subreddits.py` output)
```json
{
  "username": "Visible-Design4180",
  "total_posts": 65,
  "first_post_date": 1705337715,
  "last_post_date": 1776693493,
  "subreddits": {
    "ethz": 12,
    "gradadmissions": 8,
    "Physics": 5,
    "EPFL": 4,
    // ... other subreddits
  },
  "subreddit_count": 18
}
```

### Overlap Analysis (`analyze_overlap.py` output)

**Subreddit Overlap CSV:**
```
Subreddit,User Count,Percentage of Total Users
r/EPFL,15,16.9%
r/Switzerland,15,16.9%
r/askswitzerland,14,15.7%
```

**Pairwise Overlap CSV:**
```
Subreddit 1,Subreddit 2,Overlap Count,Overlap Percentage
r/Switzerland,r/askswitzerland,10,71.4%
r/askswitzerland,r/zurich,7,53.8%
```

## Key Findings from r/ethz Analysis

Based on analyzing 89 r/ethz users:

1. **Top overlapping subreddits:**
   - r/EPFL (16.9% of users) - Sister university in Lausanne
   - r/Switzerland (16.9%) - General Switzerland subreddit
   - r/askswitzerland (15.7%) - Q&A about Switzerland
   - r/zurich (14.6%) - City where ETH is located
   - r/UZH (13.5%) - University of Zurich

2. **Strongest pairwise overlaps:**
   - r/Switzerland & r/askswitzerland (71.4% overlap)
   - r/askswitzerland & r/zurich (53.8% overlap)
   - r/UZH & r/zurich (50.0% overlap)

3. **User diversity:**
   - Most diverse user: u/augusts8 (posts in 98 different subreddits)
   - Average subreddits per user: 10.19

## Pagination

The Arctic Shift API limits requests to 100 posts per call. To get more posts:

1. **Use `--max-posts` parameter** for automatic pagination:
   ```bash
   python reddit_scraper.py ethz --max-posts 500
   ```

2. **How pagination works:**
   - Fetches posts in batches of 100
   - Uses the `before` parameter to get older posts
   - Adds 1.5-second delays between batches
   - Stops when reaching `--max-posts` or when no more posts are available

3. **Date ranges with pagination:**
   ```bash
   # Get 1000 posts from 2024-2025
   python reddit_scraper.py ethz --max-posts 1000 --after 2024-01-01 --before 2026-01-01
   ```

## Rate Limiting

The Arctic Shift API has rate limits. The scripts include:
- 1-second delay between requests
- Automatic handling of 429 (Too Many Requests) errors
- Rate limit header monitoring
- 1.5-second delays between pagination batches

## Input Options for Cross-Subreddit Analysis

`scrape_user_subreddits.py` now supports multiple input methods:

### 1. From a Subreddit (Original Method)
```bash
python scrape_user_subreddits.py --subreddit ethz --user-limit 100
```

### 2. From User Files (Output of `scrape_users.py`)
```bash
# From CSV file
python scrape_user_subreddits.py --input ethz_users.csv --posts-per-user 200

# From JSON file
python scrape_user_subreddits.py --input ethz_users.json --after 2024-01-01
```

### 3. From Specific Users
```bash
python scrape_user_subreddits.py --users user1,user2,user3 --posts-per-user 50
```

## Date Range Filtering

All scripts now support date range filtering with `--after` and `--before` parameters:

```bash
# Analyze users active in 2024
python scrape_user_subreddits.py ethz --after 2024-01-01 --before 2025-01-01

# Analyze recent activity (last 6 months)
python scrape_user_subreddits.py ethz --after 2024-06-01

# Analyze historical activity (before 2024)
python scrape_user_subreddits.py ethz --before 2024-01-01
```

**How date filtering works:**
- **With `--subreddit`:** Finds users who posted in the target subreddit within the date range, then analyzes their posts across all subreddits within the same range
- **With `--input` or `--users`:** Analyzes all posts by the specified users, but only includes posts within the date range
- **Output:** Shows analysis limited to the specified time period

**Date formats accepted:**
- `YYYY-MM-DD` (e.g., `2024-01-01`)
- Unix epoch timestamp (e.g., `1704067200`)

## Notes

- The API only returns up to 100 posts per request
- Use `--max-posts` for pagination to get more than 100 posts
- User analysis is based on posts, not comments
- Deleted users appear as `[deleted]` and are filtered out
- The analysis covers approximately the last 6 months of activity by default
- Date range filtering allows analysis of specific time periods

## Troubleshooting

**No posts found:**
- Check if the subreddit exists
- Try without date filters
- The subreddit may have low activity

**Rate limiting:**
- The script automatically waits 60 seconds if rate limited
- Reduce `--max-posts` or `--user-limit` parameters
- Pagination adds 1.5-second delays between batches

**API errors:**
- Check your internet connection
- Verify the Arctic Shift API is accessible
- The API may be temporarily unavailable