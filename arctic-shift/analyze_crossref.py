#!/usr/bin/env python3
"""
Cross-subreddit co-occurrence analysis.
Input: CSV files produced by scrape_user_subreddits_crossref.py.

Builds a subreddit co-occurrence graph centred on the target subreddit,
runs Louvain community detection, and outputs an interactive Pyvis HTML
plus GEXF for external tools.

Usage:
  python analyze_crossref.py subreddit_scrapes/*
  python analyze_crossref.py subreddit_scrapes/* --target-sub ethz --output-dir results/
  python analyze_crossref.py subreddit_scrapes/* --min-shared-users 5 --pyvis-min-edge 20
"""

import argparse
import colorsys
import csv
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx
import networkx.algorithms.community as nx_community  # Louvain in networkx ≥3.0


@dataclass
class UserRecord:
    """One row from the crossref CSV — a user's activity across all subreddits."""

    username: str
    total_posts: int
    total_comments: int
    total_items: int  # posts + comments combined
    subreddit_count: int
    first_date: str
    last_date: str
    subreddits: Dict[str, int]  # subreddit → interaction count


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_subreddits(packed: str) -> Dict[str, int]:
    """Parse 'sub1:count1, sub2:count2' packed string into {sub: count} dict."""
    result: Dict[str, int] = {}
    if not packed or not packed.strip():
        return result
    for part in packed.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        sub, _, cnt_str = part.rpartition(":")
        result[sub.strip()] = int(cnt_str.strip())
    return result


def load_csv(path: str) -> List[UserRecord]:
    """Load one crossref CSV file into a list of UserRecords."""
    csv.field_size_limit(
        10 * 1024 * 1024
    )  # packed subreddits field can exceed default 131072
    records: List[UserRecord] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                UserRecord(
                    username=row["username"],
                    total_posts=int(row["total_posts"]),
                    total_comments=int(row["total_comments"]),
                    total_items=int(row["total_items"]),
                    subreddit_count=int(row["subreddit_count"]),
                    first_date=row["first_date"],
                    last_date=row["last_date"],
                    subreddits=_parse_subreddits(row["subreddits"]),
                )
            )
    return records


def merge_records(batches: List[List[UserRecord]]) -> List[UserRecord]:
    """Merge records across multiple CSVs — same username → combine counts additively."""
    merged: Dict[str, UserRecord] = {}
    for batch in batches:
        for r in batch:
            if r.username not in merged:
                merged[r.username] = r
                continue
            ex = merged[r.username]  # existing record for this user
            ex.total_posts += r.total_posts
            ex.total_comments += r.total_comments
            ex.total_items += r.total_items
            for sub, cnt in r.subreddits.items():
                ex.subreddits[sub] = ex.subreddits.get(sub, 0) + cnt
            ex.subreddit_count = len(ex.subreddits)
    return list(merged.values())


# ---------------------------------------------------------------------------
# Co-occurrence network
# ---------------------------------------------------------------------------


def build_cooccurrence_graph(
    records: List[UserRecord],
    target_sub: str,
    min_shared_users: int = 2,
    max_nodes: int = 300,
) -> nx.Graph:
    """
    Build a weighted subreddit co-occurrence graph centred on target_sub.
    Edge(A, B).weight = number of users active in both subreddits.
    target_sub is always included regardless of how many users it has.
    Other near-universal subreddits (>80% of users) are excluded to avoid
    trivial edges that connect everything to each other.
    """
    sub_users: Dict[str, Set[str]] = defaultdict(set)  # sub → set of usernames
    for r in records:
        for sub in r.subreddits:
            sub_users[sub].add(r.username)

    total_users = len(records)
    eligible: Set[str] = {
        sub
        for sub, users in sub_users.items()
        if sub == target_sub or (1 < len(users) < total_users * 0.8)
    }

    # Keep top max_nodes by user count; target_sub always wins a slot
    ranked = sorted(eligible, key=lambda s: len(sub_users[s]), reverse=True)
    if target_sub not in ranked[:max_nodes]:
        top_subs = [target_sub] + [s for s in ranked if s != target_sub][
            : max_nodes - 1
        ]
    else:
        top_subs = ranked[:max_nodes]
    top_set = set(top_subs)

    edge_weights: Dict[Tuple[str, str], int] = defaultdict(
        int
    )  # (a, b) → shared user count
    for r in records:
        active = [s for s in r.subreddits if s in top_set]
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                key = (min(a, b), max(a, b))
                edge_weights[key] += 1

    G = nx.Graph()
    G.add_nodes_from(top_subs)
    for (a, b), w in edge_weights.items():
        if w >= min_shared_users:
            G.add_edge(a, b, weight=w)
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
    """Run Louvain community detection (networkx ≥3.0). seed=42 for reproducibility."""
    assert G.number_of_nodes() > 0, "Graph is empty"
    community_sets = nx_community.louvain_communities(G, weight="weight", seed=42, resolution=resolution)
    partition: Dict[str, int] = {}
    for cid, nodes in enumerate(community_sets):
        for node in nodes:
            partition[node] = cid
    return partition


def compute_network_stats(G: nx.Graph, partition: Dict[str, int]) -> Dict:
    """Density, modularity, eigenvector centrality — key metrics from Volk et al."""
    assert G.number_of_edges() > 0, "Graph has no edges"
    density = nx.density(G)
    community_sets = [
        {n for n, cid in partition.items() if cid == c} for c in set(partition.values())
    ]
    modularity = nx_community.modularity(G, community_sets, weight="weight")
    try:
        eigvec = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigvec = nx.degree_centrality(G)  # fallback if power iteration doesn't converge
    top_central = sorted(eigvec.items(), key=lambda x: -x[1])[:10]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": density,
        "modularity": modularity,
        "n_communities": len(set(partition.values())),
        "top_centrality": top_central,
        "eigvec": eigvec,
    }


def community_summary(
    G: nx.Graph, partition: Dict[str, int]
) -> Dict[int, List[Tuple[str, int]]]:
    """Top-5 nodes per community ranked by internal weighted degree."""
    groups: Dict[int, List[str]] = defaultdict(list)  # cid → member nodes
    for node, cid in partition.items():
        groups[cid].append(node)
    return {
        cid: sorted(
            dict(G.subgraph(nodes).degree(weight="weight")).items(), key=lambda x: -x[1]
        )[:5]
        for cid, nodes in groups.items()
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(
    records: List[UserRecord],
    target_sub: str,
    stats: Dict,
    partition: Dict[str, int],
    G: nx.Graph,
) -> None:
    """Print network summary to stdout."""
    print(f"\n=== r/{target_sub} cross-subreddit network ===\n")
    print(f"Users: {len(records)}")

    if not stats:
        return

    print(f"\n-- Network --")
    print(f"  Nodes: {stats['n_nodes']}  Edges: {stats['n_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Modularity: {stats['modularity']:.3f}")
    print(f"  Communities: {stats['n_communities']}")

    print("\n  Top subreddits by eigenvector centrality:")
    for sub, score in stats["top_centrality"]:
        print(f"    r/{sub:<30s} {score:.4f}  (community {partition.get(sub, -1)})")

    print("\n  Communities:")
    for cid, top_nodes in sorted(community_summary(G, partition).items()):
        members = ", ".join(f"r/{s}" for s, _ in top_nodes)
        print(f"    {cid}: {members}")


def save_network_nodes_csv(
    G: nx.Graph, partition: Dict[str, int], stats: Dict, path: str
) -> None:
    """Save per-node attributes as CSV."""
    eigvec = stats["eigvec"]
    degree = dict(G.degree(weight="weight"))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subreddit",
                "community",
                "weighted_degree",
                "eigenvector_centrality",
            ],
        )
        w.writeheader()
        for node in G.nodes():
            w.writerow(
                {
                    "subreddit": node,
                    "community": partition.get(node, -1),
                    "weighted_degree": degree.get(node, 0),
                    "eigenvector_centrality": round(eigvec.get(node, 0.0), 6),
                }
            )
    print(f"Saved {G.number_of_nodes()} nodes → {path}")


def save_network_edges_csv(G: nx.Graph, path: str) -> None:
    """Save edge list as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "weight"])
        w.writeheader()
        for a, b, data in G.edges(data=True):
            w.writerow({"source": a, "target": b, "weight": data.get("weight", 1)})
    print(f"Saved {G.number_of_edges()} edges → {path}")


def _generate_colors(n: int) -> List[str]:
    """Generate n perceptually distinct hex colors by spacing hues evenly around the HSV wheel."""
    colors = []
    for i in range(n):
        h = i / n  # evenly spaced hue
        r, g, b = colorsys.hsv_to_rgb(
            h, 0.75, 0.88
        )  # fixed saturation/brightness for readability
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def _filter_by_community_size(
    partition: Dict[str, int], min_community_size: int
) -> Set[str]:
    """Return nodes whose community has at least min_community_size members."""
    community_sizes: Dict[int, int] = defaultdict(int)  # cid → member count
    for cid in partition.values():
        community_sizes[cid] += 1
    return {
        n for n, cid in partition.items() if community_sizes[cid] >= min_community_size
    }


def _filter_by_degree(
    G: nx.Graph, candidates: Set[str], min_neighbors: int
) -> Set[str]:
    """Drop nodes from candidates that have fewer than min_neighbors neighbors among candidates."""
    degree: Dict[str, int] = defaultdict(int)  # node → neighbor count within candidates
    for a, b in G.edges():
        if a in candidates and b in candidates:
            degree[a] += 1
            degree[b] += 1
    return {n for n in candidates if degree[n] >= min_neighbors}


def save_pyvis_html(
    G: nx.Graph,
    partition: Dict[str, int],
    stats: Dict,
    records: List[UserRecord],
    target_sub: str,
    path: str,
    min_edge_weight: int = 10,
    min_community_size: int = 1,  # hide nodes from communities smaller than this
    min_neighbors: int = 1,  # hide nodes with fewer visible edges than this after weight filter
) -> None:
    """
    Generate an interactive HTML network visualization using Pyvis.
    Open the output file in any browser — no server needed.
    Nodes: sized by user_count, coloured by community, target_sub highlighted.
    Edges: only those with weight >= min_edge_weight are shown.
    Communities with fewer than min_community_size subreddits are hidden entirely.
    Nodes with fewer than min_neighbors visible edges are hidden entirely.
    """
    from pyvis.network import Network

    candidates = _filter_by_community_size(partition, min_community_size)
    visible = _filter_by_degree(G, candidates, min_neighbors)

    # map only the visible communities to evenly-spaced hues — avoids red-clustering
    # when few communities survive filtering
    visible_cids = sorted({partition[n] for n in visible if n != target_sub})
    cid_color: Dict[int, str] = dict(
        zip(visible_cids, _generate_colors(len(visible_cids)))
    )

    eigvec = stats["eigvec"]
    sub_users: Dict[str, int] = defaultdict(int)  # users per subreddit for node sizing
    for r in records:
        for sub in r.subreddits:
            if sub in G:
                sub_users[sub] += 1

    max_users = max((sub_users[n] for n in G.nodes()), default=1)

    # Compute layout on the visible subgraph once — Kamada-Kawai minimises edge-length
    # variance and produces a clean static result with no simulation needed.
    visible_subgraph = G.subgraph(visible)
    pos = nx.kamada_kawai_layout(visible_subgraph, weight="weight", scale=6000)

    net = Network(
        height="95vh",
        width="100%",
        bgcolor="#0d1117",
        font_color="#e6edf3",
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",  # embed vis.js so the file works without internet/CDN
    )

    for node in G.nodes():
        if node not in visible:
            continue
        cid = partition.get(node, 0)
        user_count = sub_users[node]
        size = 10 + 50 * (user_count / max_users)
        centrality = round(eigvec.get(node, 0.0), 4)
        color = "#ffffff" if node == target_sub else cid_color.get(cid, "#888888")
        x, y = pos[node]
        net.add_node(
            node,
            label=node,
            size=size,
            color=color,
            title=f"<div>r/{node}<br>Community: {cid}<br>Users: {user_count}<br>Centrality: {centrality}</div>",
            x=x,
            y=y,
            physics=False,  # pin node at computed position — no simulation
        )

    for a, b, data in G.edges(data=True):
        if a not in visible or b not in visible:
            continue
        w = data.get("weight", 1)
        if w >= min_edge_weight:
            net.add_edge(a, b, value=w, title=f"{w} shared users")

    net.set_options("""{
      "physics": { "enabled": false },
      "edges": { "smooth": false, "color": { "opacity": 0.4 } },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }""")

    net.write_html(path)
    n_shown = sum(
        1
        for a, b, d in G.edges(data=True)
        if a in visible and b in visible and d.get("weight", 1) >= min_edge_weight
    )
    print(f"Saved Pyvis HTML ({len(visible)} nodes, {n_shown} edges shown) → {path}")


# ---------------------------------------------------------------------------
# User → community assignment
# ---------------------------------------------------------------------------


def assign_user_communities(
    records: List[UserRecord],
    partition: Dict[str, int],
    G: nx.Graph,
) -> List[Dict]:
    """
    Assign each user to a community via weighted vote across their subreddits.
    For each subreddit the user interacted with that exists in the graph, its
    community gets votes equal to the user's interaction count in that subreddit.
    The community with the highest total votes wins.
    Returns a list of dicts ready to write as CSV rows.
    """
    community_top_subs: Dict[int, str] = _community_top_subs(partition, G)
    rows = []
    for r in records:
        votes: Dict[int, int] = defaultdict(int)  # community_id → weighted vote count
        for sub, count in r.subreddits.items():
            if sub in partition:
                votes[partition[sub]] += count
        if not votes:
            community_id = -1
            community_label = "unassigned"
        else:
            community_id = max(votes, key=lambda c: votes[c])
            community_label = community_top_subs.get(community_id, "")
        rows.append(
            {
                "username": r.username,
                "community_id": community_id,
                "community_label": community_label,
                "total_items": r.total_items,
                "subreddit_count": r.subreddit_count,
            }
        )
    return rows


def _community_top_subs(
    partition: Dict[str, int], G: nx.Graph, top_n: int = 5
) -> Dict[int, str]:
    """Build a human-readable label for each community from its top-degree subreddits."""
    degree = dict(G.degree(weight="weight"))
    by_community: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for sub, cid in partition.items():
        by_community[cid].append((sub, degree.get(sub, 0)))
    return {
        cid: ", ".join(s for s, _ in sorted(members, key=lambda x: -x[1])[:top_n])
        for cid, members in by_community.items()
    }


def save_user_communities_csv(rows: List[Dict], path: str) -> None:
    """Write the user→community mapping to CSV."""
    fieldnames = [
        "username",
        "community_id",
        "community_label",
        "total_items",
        "subreddit_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} user→community assignments → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a subreddit co-occurrence graph from crossref CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --input-dir subreddit_scrapes --output-dir results/
  %(prog)s --min-shared-users 5 --pyvis-min-edge 20
        """,
    )
    parser.add_argument(
        "--input-dir",
        default="subreddit_scrapes",
        help="Directory containing crossref CSV files (default: subreddit_scrapes/)",
    )
    parser.add_argument(
        "--target-sub", default="ethz", help="Target subreddit (default: ethz)"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--min-shared-users",
        type=int,
        default=2,
        help="Min shared users for a graph edge (default: 2)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=300,
        help="Max subreddit nodes in graph (default: 300)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution — higher = more, smaller communities (default: 1.0)",
    )
    parser.add_argument(
        "--pyvis-min-edge",
        type=int,
        default=10,
        help="Min edge weight shown in Pyvis HTML (default: 10)",
    )
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=1,
        help="Hide communities with fewer than this many subreddits (default: 1)",
    )
    parser.add_argument(
        "--pyvis-min-neighbors",
        type=int,
        default=1,
        help="Hide nodes with fewer visible edges than this after weight filter (default: 1)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t = args.target_sub  # short alias used in output filenames

    csv_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    assert csv_paths, f"No CSV files found in {args.input_dir!r}"
    print(f"Found {len(csv_paths)} CSV files in {args.input_dir}/")

    batches = [load_csv(p) for p in csv_paths]
    records = merge_records(batches)
    assert records, "No records loaded — check input files"
    print(f"Loaded {len(records)} unique users from {len(csv_paths)} file(s)")

    print(
        f"Building co-occurrence graph (min_shared={args.min_shared_users}, max_nodes={args.max_nodes})..."
    )
    G = build_cooccurrence_graph(records, t, args.min_shared_users, args.max_nodes)
    assert G.number_of_edges() > 0, (
        "Graph has no edges — try lowering --min-shared-users"
    )
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = detect_communities(G, resolution=args.resolution)
    stats = compute_network_stats(G, partition)

    print_summary(records, t, stats, partition, G)

    save_network_nodes_csv(
        G, partition, stats, os.path.join(args.output_dir, f"{t}_network_nodes.csv")
    )
    save_network_edges_csv(G, os.path.join(args.output_dir, f"{t}_network_edges.csv"))
    save_pyvis_html(
        G,
        partition,
        stats,
        records,
        t,
        os.path.join(args.output_dir, f"{t}_network.html"),
        min_edge_weight=args.pyvis_min_edge,
        min_community_size=args.min_community_size,
        min_neighbors=args.pyvis_min_neighbors,
    )

    print("\nAssigning users to communities...")
    user_rows = assign_user_communities(records, partition, G)
    save_user_communities_csv(
        user_rows, os.path.join(args.output_dir, f"{t}_user_communities.csv")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
