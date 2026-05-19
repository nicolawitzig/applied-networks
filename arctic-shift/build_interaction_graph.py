"""
Build a directed weighted interaction graph from Reddit reply chains.

Nodes: users from user_voice_classification_jan_jun.csv
Edges: A → B if A's comment has parent_id pointing to a post/comment authored by B
Output: nodes CSV (voice type + community + centrality), interactive HTML
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

def load_voice_classifications(csv_path: Path) -> dict[str, dict]:
    """Map username → {voice_type, subtype, confidence} from classification CSV."""
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        result[row["username"]] = {
            "voice_type": row["voice_type"] if pd.notna(row["voice_type"]) else "unknown",
            "subtype": row["subtype"] if pd.notna(row["subtype"]) else "",
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else 0.0,
        }
    return result


def build_id_author_index(user_posts_dir: Path) -> IdAuthorIndex:
    """Scan all JSON files and build post_id/comment_id → author lookup maps."""
    index = IdAuthorIndex()
    for json_file in user_posts_dir.glob("*.json"):
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
    user_posts_dir: Path,
    index: IdAuthorIndex,
    classified_users: set[str],
) -> list[Interaction]:
    """Build edge list from reply chains — only edges touching a classified user."""
    interactions: list[Interaction] = []
    for json_file in user_posts_dir.glob("*.json"):
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

def _generate_colors(n: int) -> list[str]:
    """Generate n perceptually distinct hex colors by spacing hues around the HSV wheel."""
    colors = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
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


def save_pyvis_html(
    graph: nx.DiGraph,
    partition: dict[str, int],
    path: Path,
    min_edge_weight: int = 1,
) -> None:
    """Generate interactive directed Pyvis HTML; nodes sized by in-degree, coloured by community."""
    from pyvis.network import Network

    community_ids = sorted(set(partition.values()))
    cid_color = dict(zip(community_ids, _generate_colors(len(community_ids))))
    max_in = max((graph.in_degree(n, weight="weight") for n in graph.nodes()), default=1)

    pos = nx.kamada_kawai_layout(graph.to_undirected(), weight="weight", scale=6000)

    net = Network(
        height="95vh", width="100%",
        bgcolor="#0d1117", font_color="#e6edf3",
        directed=True, select_menu=True, filter_menu=True,
        cdn_resources="in_line",
    )

    for node in graph.nodes():
        meta = graph.nodes[node]
        cid = partition.get(node, 0)
        in_deg = graph.in_degree(node, weight="weight")
        x, y = pos[node]
        net.add_node(
            node,
            label=node,
            size=10 + 50 * (in_deg / max(max_in, 1)),
            color=cid_color.get(cid, "#888888"),
            title=f"<div>{node}<br>Voice: {meta.get('voice_type','unknown')}<br>Community: {cid}<br>In-degree: {in_deg}</div>",
            x=x, y=y, physics=False,
        )

    n_shown = 0
    for src, dst, data in graph.edges(data=True):
        w = data.get("weight", 1)
        if w >= min_edge_weight:
            net.add_edge(src, dst, value=w, title=f"{w} interactions")
            n_shown += 1

    net.set_options('{"physics":{"enabled":false},"edges":{"smooth":false,"color":{"opacity":0.4}},"interaction":{"hover":true,"tooltipDelay":100}}')
    net.write_html(str(path))
    print(f"Saved Pyvis HTML ({graph.number_of_nodes()} nodes, {n_shown} edges shown) → {path}")


# --- main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Reddit voice interaction graph")
    p.add_argument("--classifications", default="results/final/user_voice_classification_jan_jun.csv")
    p.add_argument("--posts-dir", default="results/user_posts_jan_jun")
    p.add_argument("--output-dir", default="results/final")
    p.add_argument("--min-weight", type=int, default=1, help="Min interactions for an edge to be included")
    p.add_argument("--min-edge-weight", type=int, default=1, help="Min edge weight shown in HTML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    posts_dir = Path(args.posts_dir)
    assert posts_dir.is_dir(), f"Posts directory not found: {posts_dir}"

    print("Loading voice classifications...")
    voice_map = load_voice_classifications(Path(args.classifications))
    classified_users = set(voice_map.keys())
    print(f"  {len(classified_users)} classified users")

    print("Building ID → author index...")
    index = build_id_author_index(posts_dir)
    print(f"  {len(index.posts)} posts, {len(index.comments)} comments indexed")

    print("Extracting interactions...")
    interactions = extract_interactions(posts_dir, index, classified_users)
    print(f"  {len(interactions)} raw interactions")

    print("Building interaction graph...")
    graph = build_interaction_graph(interactions, voice_map, args.min_weight)
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Detecting communities...")
    partition = detect_communities(graph)
    for node, cid in partition.items():
        graph.nodes[node]["community"] = cid

    print("Writing outputs...")
    save_nodes_csv(graph, partition, out_dir / "interaction_graph_nodes.csv")
    save_pyvis_html(graph, partition, out_dir / "interaction_graph.html", args.min_edge_weight)
    print("Done.")


if __name__ == "__main__":
    main()
