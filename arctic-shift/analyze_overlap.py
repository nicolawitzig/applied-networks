#!/usr/bin/env python3
"""
Analyze subreddit overlap from cross-subreddit analysis
"""

import json
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Set
import sys


def load_analysis(filename: str) -> List[Dict]:
    """Load analysis data from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_subreddit_overlap(analysis_data: List[Dict], 
                               exclude_subreddit: str = 'ethz',
                               min_users: int = 2) -> Dict[str, int]:
    """Calculate how many users post in each subreddit"""
    subreddit_users: Dict[str, Set[str]] = defaultdict(set)
    
    for user_data in analysis_data:
        username = user_data['username']
        subreddits = user_data['subreddits']
        
        for subreddit in subreddits.keys():
            if subreddit != exclude_subreddit:
                subreddit_users[subreddit].add(username)
    
    # Convert to counts
    overlap_counts = {sub: len(users) for sub, users in subreddit_users.items() 
                     if len(users) >= min_users}
    
    # Sort by user count (descending)
    return dict(sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True))


def calculate_pairwise_overlap(analysis_data: List[Dict], 
                              exclude_subreddit: str = 'ethz',
                              min_overlap: int = 3) -> Dict[str, Dict[str, int]]:
    """Calculate pairwise overlap between subreddits"""
    # First get user sets for each subreddit
    subreddit_users: Dict[str, Set[str]] = defaultdict(set)
    
    for user_data in analysis_data:
        username = user_data['username']
        subreddits = user_data['subreddits']
        
        for subreddit in subreddits.keys():
            if subreddit != exclude_subreddit:
                subreddit_users[subreddit].add(username)
    
    # Calculate pairwise overlaps
    pairwise_overlap: Dict[str, Dict[str, int]] = defaultdict(dict)
    subreddits = list(subreddit_users.keys())
    
    for i, sub1 in enumerate(subreddits):
        users1 = subreddit_users[sub1]
        for sub2 in subreddits[i+1:]:
            users2 = subreddit_users[sub2]
            overlap = len(users1.intersection(users2))
            if overlap >= min_overlap:
                pairwise_overlap[sub1][sub2] = overlap
                pairwise_overlap[sub2][sub1] = overlap
    
    return pairwise_overlap


def calculate_user_subreddit_matrix(analysis_data: List[Dict], 
                                   exclude_subreddit: str = 'ethz',
                                   min_users: int = 3) -> Dict[str, List[str]]:
    """Create matrix of which users are in which subreddits"""
    # Get subreddits with enough users
    overlap_counts = calculate_subreddit_overlap(analysis_data, exclude_subreddit, min_users)
    top_subreddits = list(overlap_counts.keys())
    
    # Create matrix
    matrix: Dict[str, List[str]] = defaultdict(list)
    
    for user_data in analysis_data:
        username = user_data['username']
        user_subreddits = set(user_data['subreddits'].keys())
        
        for subreddit in top_subreddits:
            if subreddit in user_subreddits:
                matrix[subreddit].append(username)
    
    return matrix


def save_overlap_analysis(overlap_counts: Dict[str, int], 
                         pairwise_overlap: Dict[str, Dict[str, int]],
                         filename_prefix: str,
                         total_users: int) -> None:
    """Save overlap analysis to files"""
    
    # Save subreddit user counts
    with open(f"{filename_prefix}_subreddit_overlap.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Subreddit', 'User Count', 'Percentage of Total Users'])
        
        for subreddit, count in overlap_counts.items():
            percentage = (count / total_users) * 100 if total_users > 0 else 0
            writer.writerow([f"r/{subreddit}", count, f"{percentage:.1f}%"])
    
    print(f"Saved subreddit overlap to {filename_prefix}_subreddit_overlap.csv")
    
    # Save pairwise overlaps
    with open(f"{filename_prefix}_pairwise_overlap.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Subreddit 1', 'Subreddit 2', 'Overlap Count', 'Overlap Percentage'])
        
        pairs_written = set()
        for sub1 in sorted(pairwise_overlap.keys()):
            for sub2, overlap in pairwise_overlap[sub1].items():
                pair_key = tuple(sorted([sub1, sub2]))
                if pair_key not in pairs_written:
                    pairs_written.add(pair_key)
                    # Calculate percentage based on smaller user base
                    user_count1 = overlap_counts.get(sub1, 0)
                    user_count2 = overlap_counts.get(sub2, 0)
                    min_users = min(user_count1, user_count2)
                    percentage = (overlap / min_users * 100) if min_users > 0 else 0
                    writer.writerow([f"r/{sub1}", f"r/{sub2}", overlap, f"{percentage:.1f}%"])
    
    print(f"Saved pairwise overlap to {filename_prefix}_pairwise_overlap.csv")


def display_overlap_analysis(overlap_counts: Dict[str, int], 
                           pairwise_overlap: Dict[str, Dict[str, int]],
                           total_users: int,
                           display_limit: int = 20) -> None:
    """Display overlap analysis in console"""
    
    print(f"\n{'='*80}")
    print(f"SUBREDDIT OVERLAP ANALYSIS")
    print(f"{'='*80}")
    print(f"Total users analyzed: {total_users}")
    print(f"Total overlapping subreddits: {len(overlap_counts)}")
    print(f"\nTop {min(display_limit, len(overlap_counts))} subreddits by user overlap:")
    print(f"{'Subreddit':<30} {'Users':<10} {'% of Total':<10}")
    print(f"{'-'*50}")
    
    for i, (subreddit, count) in enumerate(list(overlap_counts.items())[:display_limit]):
        percentage = (count / total_users) * 100
        print(f"r/{subreddit:<28} {count:<10} {percentage:.1f}%")
    
    # Find strongest pairwise overlaps
    print(f"\n{'='*80}")
    print(f"STRONGEST PAIRWISE OVERLAPS (≥3 shared users)")
    print(f"{'='*80}")
    
    all_pairs = []
    for sub1 in pairwise_overlap:
        for sub2, overlap in pairwise_overlap[sub1].items():
            pair_key = tuple(sorted([sub1, sub2]))
            if pair_key not in [p[0] for p in all_pairs]:
                all_pairs.append((pair_key, overlap))
    
    # Sort by overlap strength
    all_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Subreddit Pair':<40} {'Shared Users':<15} {'Overlap %':<15}")
    print(f"{'-'*70}")
    
    for (sub1, sub2), overlap in all_pairs[:15]:
        user_count1 = overlap_counts.get(sub1, 0)
        user_count2 = overlap_counts.get(sub2, 0)
        min_users = min(user_count1, user_count2)
        percentage = (overlap / min_users * 100) if min_users > 0 else 0
        
        print(f"r/{sub1} & r/{sub2:<32} {overlap:<15} {percentage:.1f}%")
    
    # Find subreddit clusters
    print(f"\n{'='*80}")
    print(f"SUBREDDIT CLUSTERS (≥40% overlap)")
    print(f"{'='*80}")
    
    clusters = []
    for sub1 in pairwise_overlap:
        cluster = [sub1]
        for sub2, overlap in pairwise_overlap[sub1].items():
            user_count1 = overlap_counts.get(sub1, 0)
            user_count2 = overlap_counts.get(sub2, 0)
            min_users = min(user_count1, user_count2)
            percentage = (overlap / min_users * 100) if min_users > 0 else 0
            
            if percentage >= 40:
                cluster.append(sub2)
        
        if len(cluster) > 1:
            clusters.append(sorted(set(cluster)))
    
    # Remove duplicate clusters
    unique_clusters = []
    for cluster in clusters:
        cluster_tuple = tuple(sorted(cluster))
        if cluster_tuple not in unique_clusters:
            unique_clusters.append(cluster_tuple)
    
    for i, cluster in enumerate(unique_clusters[:10]):
        print(f"Cluster {i+1}: {', '.join([f'r/{sub}' for sub in cluster])}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_overlap.py <analysis_file.json> [output_prefix]")
        print("Example: python analyze_overlap.py ethz_cross_subreddit_analysis.json ethz_overlap")
        return 1
    
    input_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "overlap_analysis"
    
    print(f"Loading analysis from {input_file}...")
    analysis_data = load_analysis(input_file)
    total_users = len(analysis_data)
    
    print(f"Analyzing {total_users} users...")
    
    # Calculate overlaps
    overlap_counts = calculate_subreddit_overlap(analysis_data, exclude_subreddit='ethz', min_users=2)
    pairwise_overlap = calculate_pairwise_overlap(analysis_data, exclude_subreddit='ethz', min_overlap=3)
    
    # Display results
    display_overlap_analysis(overlap_counts, pairwise_overlap, total_users, display_limit=25)
    
    # Save results
    save_overlap_analysis(overlap_counts, pairwise_overlap, output_prefix, total_users)
    
    print(f"\nAnalysis complete. Results saved to {output_prefix}_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())