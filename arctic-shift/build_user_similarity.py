#!/usr/bin/env python3
"""
User similarity graph and community detection.

Builds a weighted user-user graph where edge(A, B).weight = Jaccard similarity
of their subreddit sets (excluding near-universal subreddits that carry no
discriminative signal). Runs Louvain community detection on the result and
labels each community by its most characteristic subreddits (TF-IDF style).

Input:  subreddit_scrapes/*.csv  (produced by scrape_user_subreddits_crossref.py)
Output: results/user_similarity_communities.csv
        results/user_similarity_network_nodes.csv

Usage:
  python build_user_similarity.py
  python build_user_similarity.py --min-shared 3 --resolution 1.0
  python build_user_similarity.py --input-dir subreddit_scrapes --output-dir results/
"""

import argparse
import colorsys
import csv
import glob
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx
import networkx.algorithms.community as nx_community

UNIVERSAL_THRESHOLD = 0.5    # exclude subreddits used by more than this fraction of users
MIN_SUB_USERS       = 10     # exclude subreddits used by fewer than this many users


# ---------------------------------------------------------------------------
# Data loading  (same format as analyze_crossref.py)
# ---------------------------------------------------------------------------

@dataclass
class UserRecord:
    """One row from the crossref CSV."""
    username: str
    subreddits: Dict[str, int]    # subreddit → interaction count


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


def load_users(csv_paths: List[str]) -> List[UserRecord]:
    """Load and deduplicate users from one or more crossref CSVs."""
    csv.field_size_limit(10 * 1024 * 1024)
    merged: Dict[str, UserRecord] = {}
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u = row["username"].strip()
                if not u or u == "[deleted]":
                    continue
                subs = _parse_subreddits(row["subreddits"])
                if u not in merged:
                    merged[u] = UserRecord(username=u, subreddits=subs)
                else:
                    for sub, cnt in subs.items():
                        merged[u].subreddits[sub] = merged[u].subreddits.get(sub, 0) + cnt
    return list(merged.values())


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _discriminative_subs(records: List[UserRecord], min_sub_users: int = 10) -> Set[str]:
    """
    Return subreddits used by fewer than UNIVERSAL_THRESHOLD of users.
    Near-universal subreddits (r/ethz itself, r/AskReddit, etc.) are shared by
    almost everyone and carry no similarity signal — excluding them sharpens
    the community structure.
    """
    n_users = len(records)
    sub_user_count: Dict[str, int] = defaultdict(int)
    for r in records:
        for sub in r.subreddits:
            sub_user_count[sub] += 1
    return {
        sub for sub, cnt in sub_user_count.items()
        if min_sub_users <= cnt < n_users * UNIVERSAL_THRESHOLD
    }


def build_user_graph(
    records: List[UserRecord],
    min_shared: int,
    min_sub_users: int = 10,
) -> Tuple[nx.Graph, Dict[str, Set[str]]]:
    """
    Build a weighted user-user graph.

    Edge weight = Jaccard similarity of discriminative subreddit sets:
        |A ∩ B| / |A ∪ B|

    Only pairs with at least min_shared discriminative subreddits get an edge.
    Returns the graph and a mapping of username → discriminative subreddit set.
    """
    disc = _discriminative_subs(records, min_sub_users=min_sub_users)
    print(f"  {len(disc)} discriminative subreddits (excluded {sum(1 for r in records for s in r.subreddits if s not in disc) // len(records)} near-universal per user on average)")

    user_subs: Dict[str, Set[str]] = {
        r.username: {s for s in r.subreddits if s in disc}
        for r in records
    }

    # Remove users with no discriminative subreddits — they can't be placed
    user_subs = {u: subs for u, subs in user_subs.items() if subs}
    print(f"  {len(user_subs)} users have at least one discriminative subreddit")

    # Build sub → users index to enumerate pairs efficiently
    sub_users: Dict[str, List[str]] = defaultdict(list)
    for u, subs in user_subs.items():
        for sub in subs:
            sub_users[sub].append(u)

    # Accumulate shared subreddit counts per user pair
    shared_count: Dict[Tuple[str, str], int] = defaultdict(int)
    for sub, users in sub_users.items():
        if len(users) > 500:
            continue    # skip very popular discriminative subs — too many pairs
        for i, u1 in enumerate(users):
            for u2 in users[i + 1:]:
                key = (u1, u2) if u1 < u2 else (u2, u1)
                shared_count[key] += 1

    # Build graph — only edges meeting min_shared threshold
    G = nx.Graph()
    G.add_nodes_from(user_subs.keys())
    added = 0
    for (u1, u2), shared in shared_count.items():
        if shared < min_shared:
            continue
        s1, s2 = user_subs[u1], user_subs[u2]
        jaccard = shared / (len(s1) + len(s2) - shared)
        G.add_edge(u1, u2, weight=jaccard)
        added += 1

    print(f"  Graph: {G.number_of_nodes()} users, {added} edges (min_shared={min_shared})")
    return G, user_subs


# ---------------------------------------------------------------------------
# Community labelling
# ---------------------------------------------------------------------------

def label_communities(
    partition: Dict[str, int],
    user_subs: Dict[str, Set[str]],
    top_n: int = 5,
) -> Dict[int, str]:
    """
    Label each community by its most characteristic subreddits using TF-IDF logic:
    score(sub, community) = (within-community frequency) × log(n_communities / communities_containing_sub)
    This highlights subreddits that are common inside a community but rare outside it.
    """
    n_communities = len(set(partition.values()))
    community_users: Dict[int, List[str]] = defaultdict(list)
    for u, cid in partition.items():
        community_users[cid].append(u)

    # how many communities contain each subreddit (for IDF)
    sub_community_presence: Dict[str, Set[int]] = defaultdict(set)
    for cid, users in community_users.items():
        for u in users:
            for sub in user_subs.get(u, set()):
                sub_community_presence[sub].add(cid)

    labels: Dict[int, str] = {}
    for cid, users in community_users.items():
        sub_tf: Dict[str, float] = defaultdict(float)
        for u in users:
            for sub in user_subs.get(u, set()):
                sub_tf[sub] += 1.0 / len(users)    # term frequency within community

        scores: Dict[str, float] = {}
        for sub, tf in sub_tf.items():
            idf = math.log(n_communities / len(sub_community_presence[sub]))
            scores[sub] = tf * idf

        top = sorted(scores, key=lambda s: -scores[s])[:top_n]
        labels[cid] = ", ".join(top) if top else "(unlabelled)"
    return labels


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_communities_csv(
    records: List[UserRecord],
    partition: Dict[str, int],
    labels: Dict[int, str],
    path: str,
    min_community_size: int = 10,
) -> None:
    """Write one row per user: username, community_id, community_label, n_subreddits.
    Communities below min_community_size are labelled 'small_community'."""
    community_sizes: Dict[int, int] = defaultdict(int)
    for cid in partition.values():
        community_sizes[cid] += 1

    fieldnames = ["username", "community_id", "community_label", "n_subreddits"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            cid = partition.get(r.username, -1)
            if cid == -1:
                label = "unassigned"
            elif community_sizes[cid] < min_community_size:
                label = "small_community"
            else:
                label = labels.get(cid, "unassigned")
            w.writerow({
                "username":        r.username,
                "community_id":    cid,
                "community_label": label,
                "n_subreddits":    len(r.subreddits),
            })
    print(f"Saved {len(records)} users → {path}")


def save_network_nodes_csv(
    G: nx.Graph,
    partition: Dict[str, int],
    labels: Dict[int, str],
    path: str,
) -> None:
    """Write per-user graph attributes for external tools."""
    degree = dict(G.degree(weight="weight"))
    fieldnames = ["username", "community_id", "community_label", "weighted_degree"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for node in G.nodes():
            cid = partition.get(node, -1)
            w.writerow({
                "username":        node,
                "community_id":    cid,
                "community_label": labels.get(cid, "unassigned"),
                "weighted_degree": round(degree.get(node, 0.0), 4),
            })
    print(f"Saved {G.number_of_nodes()} nodes → {path}")


def print_summary(
    partition: Dict[str, int],
    labels: Dict[int, str],
    G: nx.Graph,
) -> None:
    """Print community size table to stdout."""
    community_sizes: Dict[int, int] = defaultdict(int)
    for cid in partition.values():
        community_sizes[cid] += 1

    n_communities = len(community_sizes)
    modularity = nx_community.modularity(
        G,
        [{u for u, c in partition.items() if c == cid} for cid in set(partition.values())],
        weight="weight",
    )

    print(f"\n=== User similarity communities ===")
    print(f"  Users in graph: {G.number_of_nodes()}")
    print(f"  Communities:    {n_communities}")
    print(f"  Modularity:     {modularity:.3f}\n")
    print(f"  {'cid':<5}  {'users':<7}  label")
    print(f"  {'-'*5}  {'-'*7}  {'-'*50}")
    for cid, size in sorted(community_sizes.items(), key=lambda x: -x[1]):
        print(f"  {cid:<5}  {size:<7}  {labels.get(cid, '')}")


# ---------------------------------------------------------------------------
# Pyvis visualisation
# ---------------------------------------------------------------------------

def _generate_colors(n: int) -> List[str]:
    """Generate n perceptually distinct hex colors evenly spaced around the HSV wheel."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def save_pyvis_html(
    G: nx.Graph,
    partition: Dict[str, int],
    labels: Dict[int, str],
    path: str,
    top_communities: int = 12,
    users_per_community: int = 40,
    min_community_size: int = 10,
    seed: int = 42,
) -> None:
    """
    Visualise a sampled subgraph: take the top_communities largest communities
    (each with at least min_community_size members), sample up to
    users_per_community users from each, compute a static Kamada-Kawai layout,
    and write a self-contained Pyvis HTML.
    """
    from pyvis.network import Network

    random.seed(seed)

    # Pick top communities by size, excluding those below the minimum size
    community_sizes: Dict[int, int] = defaultdict(int)
    for cid in partition.values():
        community_sizes[cid] += 1
    top_cids = {cid for cid, _ in sorted(community_sizes.items(), key=lambda x: -x[1])[:top_communities]
                if community_sizes[cid] >= min_community_size}

    # Sample users from each kept community
    community_members: Dict[int, List[str]] = defaultdict(list)
    for u, cid in partition.items():
        if cid in top_cids:
            community_members[cid].append(u)
    sampled: Set[str] = set()
    for cid, members in community_members.items():
        sampled.update(random.sample(members, min(users_per_community, len(members))))

    subgraph = G.subgraph(sampled)
    colors   = _generate_colors(len(top_cids))
    cid_color: Dict[int, str] = {cid: colors[i] for i, cid in enumerate(sorted(top_cids))}

    print(f"Computing layout for {subgraph.number_of_nodes()} sampled nodes...")
    pos = nx.kamada_kawai_layout(subgraph, weight="weight", scale=6000)

    degree = dict(subgraph.degree(weight="weight"))
    max_deg = max(degree.values(), default=1)

    net = Network(
        height="95vh", width="100%",
        bgcolor="#0d1117", font_color="#e6edf3",
        select_menu=True, filter_menu=True,
        cdn_resources="in_line",
    )

    for node in subgraph.nodes():
        cid  = partition.get(node, -1)
        size = 8 + 30 * (degree.get(node, 0) / max_deg)
        x, y = pos[node]
        net.add_node(
            node,
            label=node,
            size=size,
            color=cid_color.get(cid, "#888888"),
            title=f"<div>u/{node}<br>Community {cid}:<br>{labels.get(cid, '')}</div>",
            x=x, y=y,
            physics=False,
        )

    for u, v, data in subgraph.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 0.1), title=f"Jaccard: {data.get('weight', 0):.3f}")

    net.set_options("""{
      "physics": { "enabled": false },
      "edges": { "smooth": false, "color": { "opacity": 0.3 } },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }""")

    net.write_html(path)
    print(f"Saved Pyvis HTML ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges) → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a user similarity graph and detect communities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --min-shared 3 --resolution 1.0
  %(prog)s --input-dir subreddit_scrapes --output-dir results/
        """,
    )
    parser.add_argument("--input-dir",  default="subreddit_scrapes",
                        help="Directory containing crossref CSVs (default: subreddit_scrapes/)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--min-sub-users", type=int, default=10,
                        help="Min users a subreddit must have to count as discriminative (default: 10)")
    parser.add_argument("--min-shared", type=int, default=3,
                        help="Min shared discriminative subreddits for an edge (default: 3)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution — higher = more, smaller communities (default: 1.0)")
    parser.add_argument("--top-communities", type=int, default=12,
                        help="Number of largest communities to include in visualisation (default: 12)")
    parser.add_argument("--users-per-community", type=int, default=40,
                        help="Max users sampled per community for visualisation (default: 40)")
    parser.add_argument("--min-community-size", type=int, default=10,
                        help="Communities smaller than this are labelled 'small_community' in output "
                             "and excluded from visualisation (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    assert csv_paths, f"No CSV files found in {args.input_dir!r}"
    print(f"Found {len(csv_paths)} CSV files in {args.input_dir}/")

    records = load_users(csv_paths)
    assert records, "No users loaded"
    print(f"Loaded {len(records)} unique users")

    print("Building user similarity graph...")
    G, user_subs = build_user_graph(records, min_shared=args.min_shared,
                                    min_sub_users=args.min_sub_users)
    assert G.number_of_edges() > 0, "No edges — try lowering --min-shared"

    print("Running Louvain community detection...")
    community_sets = nx_community.louvain_communities(G, weight="weight", seed=42, resolution=args.resolution)
    partition: Dict[str, int] = {}
    for cid, nodes in enumerate(community_sets):
        for node in nodes:
            partition[node] = cid
    print(f"  {len(community_sets)} communities found")

    labels = label_communities(partition, user_subs)
    print_summary(partition, labels, G)

    save_communities_csv(records, partition, labels,
                         os.path.join(args.output_dir, "user_similarity_communities.csv"),
                         min_community_size=args.min_community_size)
    save_network_nodes_csv(G, partition, labels,
                           os.path.join(args.output_dir, "user_similarity_network_nodes.csv"))
    save_pyvis_html(G, partition, labels,
                    os.path.join(args.output_dir, "user_similarity_network.html"),
                    top_communities=args.top_communities,
                    users_per_community=args.users_per_community,
                    min_community_size=args.min_community_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
