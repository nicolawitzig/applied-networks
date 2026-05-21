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
    detect_communities,
    _to_undirected_weighted
)
import networkx as nx
from collections import defaultdict

def main():
    csv_paths = [Path("results/final/user_voice_classifications.csv")]
    posts_dirs = [Path("results/user_posts")]

    print("Loading classifications...")
    voice_map = load_voice_classifications(csv_paths)
    classified_users = set(voice_map.keys())

    print("Building author index...")
    index = build_id_author_index(posts_dirs)

    print("Extracting interactions...")
    replies = extract_interactions(posts_dirs, index, classified_users)
    mentions = extract_mention_interactions(posts_dirs, classified_users)
    post_commenters = build_post_commenters_index(posts_dirs)
    coparticipants = extract_coparticipant_interactions(post_commenters, classified_users)

    interactions = replies + mentions + coparticipants
    print(f"Found {len(interactions)} total interactions")

    graph = build_interaction_graph(interactions, voice_map, min_weight=1)
    graph = filter_isolated_unknowns(graph)
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

    if graph.number_of_nodes() == 0:
        print("Empty graph.")
        return

    print("Detecting communities...")
    partition = detect_communities(graph)
    print(f"Found {len(set(partition.values()))} communities")

    undirected = _to_undirected_weighted(graph)
    communities = [{n for n, c in partition.items() if c == cid} for cid in set(partition.values())]
    
    try:
        modularity = nx.algorithms.community.modularity(undirected, communities, weight="weight")
        print(f"Modularity: {modularity:.4f}")
    except Exception as e:
        print(f"Error calculating modularity: {e}")

    intra_weight = 0
    inter_weight = 0
    for u, v, data in undirected.edges(data=True):
        w = data.get("weight", 1)
        if partition[u] == partition[v]:
            intra_weight += w
        else:
            inter_weight += w

    total_weight = intra_weight + inter_weight
    if total_weight > 0:
        print(f"Intra-community weight: {intra_weight} ({(intra_weight/total_weight)*100:.1f}%)")
        print(f"Inter-community weight: {inter_weight} ({(inter_weight/total_weight)*100:.1f}%)")
    
    comm_sizes = defaultdict(int)
    for n, c in partition.items():
        comm_sizes[c] += 1

    print("\nTop 10 communities by size:")
    for c, size in sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Community {c}: {size} nodes")

if __name__ == '__main__':
    main()
