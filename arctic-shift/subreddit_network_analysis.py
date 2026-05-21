import networkx as nx
from collections import Counter
from pathlib import Path
from build_subreddit_graph import (
    load_voice_classifications,
    build_subreddit_data,
    build_subreddit_graph,
    detect_communities,
    get_community_voice_distribution
)

def main():
    csv_paths = [Path("results/final/user_voice_classifications.csv")]
    posts_dirs = [Path("results/user_posts")]

    print("Loading classifications...")
    voice_map = load_voice_classifications(csv_paths)
    
    print("Building subreddit data...")
    sub_voice, user_subs = build_subreddit_data(posts_dirs, voice_map)
    
    print("Building graph...")
    # Using defaults from build_subreddit_graph.py
    graph = build_subreddit_graph(sub_voice, user_subs, min_weight=2, min_users=2)
    
    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    
    print("Detecting communities...")
    partition = detect_communities(graph, resolution=1.0)
    
    # Remove singletons
    singleton_cids = {cid for cid, cnt in Counter(partition.values()).items() if cnt == 1}
    singleton_nodes = [n for n, cid in partition.items() if cid in singleton_cids]
    graph.remove_nodes_from(singleton_nodes)
    partition = {n: cid for n, cid in partition.items() if n in graph}
    
    with open("results/subreddit_analysis.txt", "w") as f:
        f.write("=== SUBREDDIT NETWORK ANALYSIS ===\n")
        f.write(f"Nodes (Subreddits): {graph.number_of_nodes()}\n")
        f.write(f"Edges (Shared user connections): {graph.number_of_edges()}\n\n")
        
        f.write("--- DEGREE CENTRALITY (Top Subreddits by Connections) ---\n")
        degrees = sorted(graph.degree(weight="weight"), key=lambda x: x[1], reverse=True)
        for sub, deg in degrees[:10]:
            dom_voice = graph.nodes[sub].get('dominant_voice_type', 'unknown')
            users = graph.nodes[sub].get('user_count', 0)
            f.write(f"  {sub}: weight {deg}, {users} users (Dominant: {dom_voice})\n")
            
        f.write("\n--- PAGERANK (Most Influential Subreddits) ---\n")
        pr = nx.pagerank(graph, weight="weight")
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        for sub, p in sorted_pr[:10]:
            f.write(f"  {sub}: {p:.4f}\n")
            
        f.write("\n--- COMMUNITY STRUCTURE ---\n")
        communities = [{n for n, c in partition.items() if c == cid} for cid in set(partition.values())]
        try:
            modularity = nx.algorithms.community.modularity(graph, communities, weight="weight")
            f.write(f"Modularity: {modularity:.4f}\n")
        except Exception as e:
            f.write(f"Modularity: Error ({e})\n")
            
        f.write(f"Number of valid communities: {len(communities)}\n")
        
        community_voice = get_community_voice_distribution(graph, partition)
        sizes = Counter(partition.values())
        
        for cid, size in sizes.most_common():
            f.write(f"\nCommunity {cid} ({size} subreddits):\n")
            f.write(f"  Voice distribution: {community_voice.get(cid, 'N/A')}\n")
            
            # Top subreddits in this community by pagerank
            comm_nodes = [n for n, c in partition.items() if c == cid]
            top_nodes = sorted(comm_nodes, key=lambda n: pr[n], reverse=True)[:10]
            f.write(f"  Top subreddits: {', '.join(top_nodes)}\n")
            
    print("Analysis saved to results/subreddit_analysis.txt")

if __name__ == '__main__':
    main()
