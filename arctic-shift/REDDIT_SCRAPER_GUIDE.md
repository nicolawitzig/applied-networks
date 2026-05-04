## Analysis Method

The analysis pipeline adapts the network approach from Blondel et al. (2008) and follows a three-stage design:

### Stage 1: User Discovery (Exhaustive Pagination)

**Input:** target subreddit name + date range (e.g., `--subreddit ethz --after 2024-01-01 --before 2026-01-01`).  
**Process:** `scrape_user_subreddits_crossref.py` exhaustively paginates all posts and comments in the target subreddit using the Arctic Shift API (100 items per request, cursor-based via the `before` parameter). Every unique author with a non-deleted username is collected.  
**Output:** a set of all users who posted or commented in the target subreddit within the date range.

### Stage 2: Cross-Referencing

**Input:** the set of discovered users from Stage 1.  
**Process:** for each user, the script paginates all of their posts and comments across *all* subreddits within the same date range. Each user's activity is aggregated into a `UserActivity` record: which subreddits they participate in, how many posts/comments per subreddit, and their first/last activity timestamps.  
**Output:** a CSV with columns `username`, `total_posts`, `total_comments`, `total_items`, `subreddit_count`, `first_date`, `last_date`, and `subreddits` (packed `sub:count` pairs).

### Stage 3: Overlap + Network Analysis

**Input:** the CSV from Stage 2.  
**Process (two parallel analyses):**

1. **Overlap analysis** — counts how many users share a given subreddit (excluding the target subreddit) and computes pairwise subreddit overlap (number of users active in both subreddits). Results are saved as CSV tables.

2. **Network analysis (Louvain community detection)** — builds a weighted undirected graph where:
   - **Nodes** = subreddits (excluding the target)
   - **Edge weights** = number of shared users between two subreddits
   - **Community detection** via the Louvain algorithm (Blondel et al., *J. Stat. Mech.*, 2008), implemented in pure Python (no external dependencies)
   - **Modularity** computed to measure community structure quality
   - **Network density** calculated as realized edges / max possible edges

**Outputs:**

| File | Description |
|---|---|
| `{prefix}_subreddit_overlap.csv` | Per-subreddit user counts and percentage of total users |
| `{prefix}_pairwise_overlap.csv` | Pairwise subreddit overlap counts and percentages |
| `{prefix}_communities.csv` | Louvain community assignments for each subreddit |
| `{prefix}_network.gexf` | GEXF graph file importable into **Gephi** for visualization |

Console output at Stage 3 includes: number of nodes, edges, network density, modularity score, number of detected communities, and representative (highest-degree) subreddits per community.

### Relationship to Prior Work

This method mirrors the approach used by Fürst et al. (2022) and Sörensen et al. (2023) in their analysis of university-affiliated Twitter accounts. Where they used @mention networks and Louvain community detection on Twitter, we substitute subreddit co-occurrence as the relational tie — two subreddits are connected if they share a critical mass of users. The Louvain algorithm is identical, and the GEXF output enables identical visualization workflows in Gephi.

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