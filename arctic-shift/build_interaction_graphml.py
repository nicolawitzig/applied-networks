"""
Build a directed weighted interaction graph from Reddit reply chains.

Nodes: users from user_voice_classification_jan_jun.csv
Edges: A → B if A's comment has parent_id pointing to a post/comment authored by B
Output: nodes CSV (voice type + community + centrality), GEXF network file
"""
from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import community as community_louvain
import networkx as nx
import pandas as pd


# --- data types ---

@dataclass
class Interaction:
    src: str       # replying user
    dst: str       # user being replied to
    subreddit: str


@dataclass
class IdAuthorIndex:
    posts: dict[str, str] = field(default_factory=dict)       # post_id → author
    comments: dict[str, str] = field(default_factory=dict)    # comment_id → author
    comment_sub: dict[str, str] = field(default_factory=dict) # comment_id → subreddit


# --- steps ---

def load_voice_classifications(csv_paths: list[Path]) -> dict[str, dict]:
    """Merge classification CSVs; for duplicate usernames keep the higher-confidence entry."""
    existing = [p for p in csv_paths if p.exists()]
    for p in csv_paths:
        if not p.exists():
            print(f"  Warning: classification file not found, skipping: {p}")
    assert existing, f"No classification files found among: {csv_paths}"
    df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    df = df.sort_values("confidence", ascending=False).drop_duplicates("username", keep="first")
    result = {}
    for _, row in df.iterrows():
        result[row["username"]] = {
            "voice_type": row["voice_type"] if pd.notna(row["voice_type"]) else "unknown",
            "subtype": row["subtype"] if pd.notna(row["subtype"]) else "",
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else 0.0,
        }
    return result


def build_id_author_index(user_posts_dirs: list[Path]) -> IdAuthorIndex:
    """Scan all JSON files across all posts directories and build post_id/comment_id → author lookup maps."""
    index = IdAuthorIndex()
    for posts_dir in user_posts_dirs:
        for json_file in posts_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            for post in data.get("posts", []):
                author = post.get("author", "")
                if author and author != "[deleted]":
                    index.posts[post["id"]] = author
            for comment in data.get("comments", []):
                author = comment.get("author", "")
                if author and author != "[deleted]":
                    index.comments[comment["id"]] = author
                    index.comment_sub[comment["id"]] = comment.get("subreddit", "")
    return index


def _resolve_parent(parent_id: str, post_index: dict, comment_index: dict, comment_sub: dict) -> tuple[str, str] | None:
    """Return (author, subreddit) for a parent_id, or None if not resolvable."""
    assert parent_id.startswith("t1_") or parent_id.startswith("t3_"), \
        f"Unexpected parent_id format: {parent_id!r}"
    raw_id = parent_id[3:]
    if parent_id.startswith("t1_"):
        author = comment_index.get(raw_id)
        subreddit = comment_sub.get(raw_id, "")
        return (author, subreddit) if author else None
    author = post_index.get(raw_id)
    return (author, "") if author else None


def extract_interactions(
    user_posts_dirs: list[Path],
    index: IdAuthorIndex,
    classified_users: set[str],
) -> list[Interaction]:
    """Build edge list from reply chains across all posts directories — only edges touching a classified user."""
    interactions: list[Interaction] = []
    for posts_dir in user_posts_dirs:
        for json_file in posts_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            src = data.get("username", "")
            if not src:
                continue
            for comment in data.get("comments", []):
                parent_id = comment.get("parent_id", "")
                if not parent_id:
                    continue
                resolved = _resolve_parent(parent_id, index.posts, index.comments, index.comment_sub)
                if resolved is None:
                    continue
                dst, subreddit = resolved
                if not subreddit:
                    subreddit = comment.get("subreddit", "")
                if src == dst:
                    continue
                if src not in classified_users and dst not in classified_users:
                    continue
                interactions.append(Interaction(src=src, dst=dst, subreddit=subreddit))
    return interactions


def build_post_commenters_index(user_posts_dirs: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """Map post_id → [(commenter, subreddit), ...] by scanning link_id on every comment."""
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for posts_dir in user_posts_dirs:
        for json_file in posts_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            for comment in data.get("comments", []):
                author = comment.get("author", "")
                if not author or author == "[deleted]":
                    continue
                link_id = comment.get("link_id", "")
                if link_id.startswith("t3_"):
                    post_id = link_id[3:]
                elif comment.get("parent_id", "").startswith("t3_"):
                    post_id = comment["parent_id"][3:]
                else:
                    continue
                index[post_id].append((author, comment.get("subreddit", "")))
    return index


def extract_coparticipant_interactions(
    post_commenters: dict[str, list[tuple[str, str]]],
    classified_users: set[str],
) -> list[Interaction]:
    """Create bidirectional edges between every pair of users who commented on the same post."""
    interactions: list[Interaction] = []
    for commenters in post_commenters.values():
        seen: dict[str, str] = {}  # deduplicate per post: author → subreddit
        for author, subreddit in commenters:
            if author not in seen:
                seen[author] = subreddit
        users = list(seen.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                a, b = users[i], users[j]
                if a not in classified_users and b not in classified_users:
                    continue
                sr = seen[a] or seen[b]
                interactions.append(Interaction(src=a, dst=b, subreddit=sr))
                interactions.append(Interaction(src=b, dst=a, subreddit=sr))
    return interactions


def build_interaction_graph(
    interactions: list[Interaction],
    voice_map: dict[str, dict],
    min_weight: int = 1,
) -> nx.DiGraph:
    """Build weighted DiGraph; nodes carry voice_type/subtype/confidence attributes."""
    edge_counter: Counter = Counter()
    edge_subreddits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ix in interactions:
        edge_counter[(ix.src, ix.dst)] += 1
        edge_subreddits[(ix.src, ix.dst)].add(ix.subreddit)

    g = nx.DiGraph()
    for (src, dst), weight in edge_counter.items():
        if weight < min_weight:
            continue
        for node in (src, dst):
            if node not in g:
                meta = voice_map.get(node, {"voice_type": "unknown", "subtype": "", "confidence": 0.0})
                g.add_node(node, **meta)
        g.add_edge(src, dst, weight=weight, subreddits=",".join(sorted(edge_subreddits[(src, dst)])))
    return g


def filter_isolated_unknowns(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove unknown-voice nodes that have no edge touching a non-unknown node."""
    to_remove = [
        node for node in graph.nodes()
        if graph.nodes[node].get("voice_type") == "unknown"
        and all(
            graph.nodes[n].get("voice_type") == "unknown"
            for n in set(graph.predecessors(node)) | set(graph.successors(node))
        )
    ]
    g = graph.copy()
    g.remove_nodes_from(to_remove)
    return g


def detect_communities(graph: nx.DiGraph, resolution: float = 1.0, random_state: int = 42) -> dict[str, int]:
    """Run Louvain on undirected projection, return handle → community_id mapping."""
    undirected = graph.to_undirected()
    for u, v, d in graph.edges(data=True):
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] = undirected[u][v].get("weight", 0) + d.get("weight", 1)
    return community_louvain.best_partition(
        undirected, weight="weight", resolution=resolution, random_state=random_state
    )


# --- output ---

def _generate_rgb_colors(n: int) -> list[dict]:
    """Generate n perceptually distinct RGB dicts."""
    colors = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        colors.append({"r": int(r*255), "g": int(g*255), "b": int(b*255), "a": 1.0})
    return colors


def save_nodes_csv(graph: nx.DiGraph, partition: dict[str, int], path: Path) -> None:
    """Write per-node attributes — voice type, community, centrality — as CSV."""
    try:
        centrality = nx.eigenvector_centrality(graph.to_undirected(), weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.degree_centrality(graph)
    rows = []
    for node in graph.nodes():
        meta = graph.nodes[node]
        rows.append({
            "username": node,
            "voice_type": meta.get("voice_type", "unknown"),
            "subtype": meta.get("subtype", ""),
            "confidence": meta.get("confidence", 0.0),
            "community": partition.get(node, -1),
            "in_degree": graph.in_degree(node, weight="weight"),
            "out_degree": graph.out_degree(node, weight="weight"),
            "eigenvector_centrality": round(centrality.get(node, 0.0), 6),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} nodes → {path}")


def save_gexf(graph: nx.DiGraph, partition: dict[str, int], path: Path) -> None:
    """Export to GEXF using networkx, appending viz tags for immediate nice appearance in Gephi."""
    print("Computing visual attributes (colors/sizes) for Gephi...")
    voice_types = sorted(set(graph.nodes[n].get("voice_type", "unknown") for n in graph.nodes()))
    vt_color = dict(zip(voice_types, _generate_rgb_colors(len(voice_types))))

    max_in = max((graph.in_degree(n, weight="weight") for n in graph.nodes()), default=1)

    for node in graph.nodes():
        meta = graph.nodes[node]
        vt = meta.get("voice_type", "unknown")
        in_deg = graph.in_degree(node, weight="weight")
        size = 10.0 + 50.0 * (in_deg / max(max_in, 1))

        color = vt_color.get(vt, {"r": 128, "g": 128, "b": 128, "a": 1.0})

        meta["viz"] = {
            "color": color,
            "size": size
        }
        
    nx.write_gexf(graph, path)
    print(f"Saved GEXF network ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges) → {path}")


# --- main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Reddit voice interaction graph (GEXF format for Gephi)")
    p.add_argument("--classifications", nargs="+", default=[
        "results/final/user_voice_classification_jan_jun.csv",
        "results/final/user_voice_classification_jul_dec.csv",
    ])
    p.add_argument("--posts-dir", nargs="+", default=[
        "results/user_posts_jan_jun",
        "results/user_posts_jul_dec",
    ])
    p.add_argument("--output-dir", default="results/final")
    p.add_argument("--min-weight", type=int, default=1, help="Min interactions for an edge to be included")
    p.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution — higher = more smaller communities")
    p.add_argument("--include-unknown", action="store_true", default=False, help="Include unknown-voice users that interact with at least one classified user")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    posts_dirs = [Path(d) for d in args.posts_dir if Path(d).is_dir()]
    for d in args.posts_dir:
        if not Path(d).is_dir():
            print(f"  Warning: posts directory not found, skipping: {d}")
    assert posts_dirs, f"No posts directories found among: {args.posts_dir}"

    print("Loading voice classifications...")
    voice_map = load_voice_classifications([Path(p) for p in args.classifications])
    classified_users = set(voice_map.keys())
    print(f"  {len(classified_users)} classified users")

    print("Building ID → author index...")
    index = build_id_author_index(posts_dirs)
    print(f"  {len(index.posts)} posts, {len(index.comments)} comments indexed")

    print("Extracting direct reply interactions...")
    interactions = extract_interactions(posts_dirs, index, classified_users)
    print(f"  {len(interactions)} direct reply interactions")

    print("Extracting co-participation interactions...")
    post_commenters = build_post_commenters_index(posts_dirs)
    copart = extract_coparticipant_interactions(post_commenters, classified_users)
    interactions += copart
    print(f"  {len(copart)} co-participation interactions ({len(interactions)} total)")

    print("Building interaction graph...")
    graph = build_interaction_graph(interactions, voice_map, args.min_weight)
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if args.include_unknown:
        print("Filtering isolated unknowns...")
        graph = filter_isolated_unknowns(graph)
        print(f"  {graph.number_of_nodes()} nodes after removing unknowns with no non-unknown neighbors")
    else:
        unknown_nodes = [n for n in graph.nodes() if graph.nodes[n].get("voice_type") == "unknown"]
        graph.remove_nodes_from(unknown_nodes)
        print(f"  Removed {len(unknown_nodes)} unknown-voice nodes ({graph.number_of_nodes()} remaining)")

    print("Detecting communities...")
    partition = detect_communities(graph, resolution=args.resolution)

    singleton_cids = {cid for cid, count in Counter(partition.values()).items() if count == 1}
    singleton_nodes = [n for n, cid in partition.items() if cid in singleton_cids]
    graph.remove_nodes_from(singleton_nodes)
    partition = {n: cid for n, cid in partition.items() if n in graph}
    print(f"  Removed {len(singleton_nodes)} singleton-community nodes ({graph.number_of_nodes()} remaining)")

    for node, cid in partition.items():
        graph.nodes[node]["community"] = cid

    print("Writing outputs...")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nodes_csv(graph, partition, out_dir / "interaction_graph_nodes.csv")
    
    # Save as GEXF instead of GraphML to support visual attributes in Gephi natively
    save_gexf(graph, partition, out_dir / "interaction_graph.gexf")
    print("Done.")


if __name__ == "__main__":
    main()
