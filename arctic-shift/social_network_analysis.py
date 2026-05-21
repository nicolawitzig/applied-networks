import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from build_interaction_graph import (
    load_voice_classifications,
    build_id_author_index,
    extract_interactions,
    extract_mention_interactions,
    build_post_commenters_index,
    extract_coparticipant_interactions,
    build_interaction_graph,
    filter_isolated_unknowns,
    _to_undirected_weighted
)

def main():
    csv_paths = [Path("results/final/user_voice_classifications.csv")]
    posts_dirs = [Path("results/user_posts")]

    print("Loading data...")
    voice_map = load_voice_classifications(csv_paths)
    classified_users = set(voice_map.keys())

    index = build_id_author_index(posts_dirs)
    replies = extract_interactions(posts_dirs, index, classified_users)
    mentions = extract_mention_interactions(posts_dirs, classified_users)
    post_commenters = build_post_commenters_index(posts_dirs)
    coparticipants = extract_coparticipant_interactions(post_commenters, classified_users)

    interactions = replies + mentions + coparticipants
    
    # Track subreddits
    subreddit_interactions = Counter()
    for ix in interactions:
        subreddit_interactions[ix.subreddit] += 1
        
    print(f"Total interactions: {len(interactions)}")

    graph = build_interaction_graph(interactions, voice_map, min_weight=1)
    graph = filter_isolated_unknowns(graph)
    
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

    # 1. Voice Type Composition
    print("\n--- VOICE TYPE COMPOSITION ---")
    voice_counts = Counter(nx.get_node_attributes(graph, 'voice_type').values())
    for vt, count in voice_counts.most_common():
        print(f"{vt}: {count}")

    # 2. Degree Analysis (Power Law / Super Spreaders)
    print("\n--- DEGREE ANALYSIS ---")
    in_degrees = dict(graph.in_degree(weight='weight'))
    out_degrees = dict(graph.out_degree(weight='weight'))
    
    sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 users by In-Degree (Most replied to / mentioned):")
    for u, d in sorted_in[:5]:
        print(f"  {u} (Voice: {graph.nodes[u].get('voice_type')}): {d}")
        
    sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 users by Out-Degree (Most active repliers):")
    for u, d in sorted_out[:5]:
        print(f"  {u} (Voice: {graph.nodes[u].get('voice_type')}): {d}")

    # 3. Centrality (PageRank) - Who is most influential?
    print("\n--- INFLUENCE (PAGERANK) ---")
    pr = nx.pagerank(graph, weight='weight')
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 users by PageRank:")
    for u, p in sorted_pr[:5]:
        print(f"  {u} (Voice: {graph.nodes[u].get('voice_type')}): {p:.4f}")

    # Centrality by voice type
    vt_pr = defaultdict(list)
    for u, p in pr.items():
        vt_pr[graph.nodes[u].get('voice_type')].append(p)
    print("Average PageRank by Voice Type:")
    for vt, prs in vt_pr.items():
        if len(prs) > 5:
            print(f"  {vt}: {np.mean(prs):.6f}")

    # 4. Homophily / Assortativity
    print("\n--- HOMOPHILY ---")
    # Does like interact with like?
    try:
        assortativity = nx.attribute_assortativity_coefficient(graph, 'voice_type')
        print(f"Voice Type Assortativity Coefficient: {assortativity:.4f}")
    except Exception as e:
        print(f"Could not calculate assortativity: {e}")

    # 5. Subreddit Activity
    print("\n--- SUBREDDIT ACTIVITY ---")
    print("Top 10 subreddits driving interactions:")
    for sub, count in subreddit_interactions.most_common(10):
        print(f"  {sub}: {count}")

if __name__ == '__main__':
    main()
