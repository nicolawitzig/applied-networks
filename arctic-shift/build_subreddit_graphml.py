"""
Build a subreddit co-participation graph and export to GEXF for Gephi.

Nodes: subreddits with at least --min-users classified (non-unknown) users active in them
Edges: A — B weighted by number of classified users active in both subreddits
Output: nodes CSV + GEXF with viz colors/sizes baked in for immediate Gephi use
"""

from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter, defaultdict
from pathlib import Path

import community as community_louvain
import networkx as nx
import pandas as pd


# --- data loading ---


def load_voice_classifications(csv_paths: list[Path]) -> dict[str, dict]:
    """Merge classification CSVs; for duplicate usernames keep the higher-confidence entry."""
    existing = [p for p in csv_paths if p.exists()]
    for p in csv_paths:
        if not p.exists():
            print(f"  Warning: classification file not found, skipping: {p}")
    assert existing, f"No classification files found among: {csv_paths}"
    df = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    df = df.sort_values("confidence", ascending=False).drop_duplicates(
        "username", keep="first"
    )
    result = {}
    for _, row in df.iterrows():
        result[row["username"]] = {
            "voice_type": row["voice_type"] if pd.notna(row["voice_type"]) else "unknown",
            "subtype": row["subtype"] if pd.notna(row["subtype"]) else "",
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else 0.0,
        }
    return result


def build_subreddit_data(
    user_posts_dirs: list[Path],
    voice_map: dict[str, dict],
) -> tuple[dict[str, Counter], dict[str, set[str]]]:
    """Single pass over user JSONs; return (subreddit→voice_counter, user→subreddits). Skips unknown-voice users."""
    sub_voice: dict[str, Counter] = defaultdict(Counter)
    user_subs: dict[str, set[str]] = defaultdict(set)
    for posts_dir in user_posts_dirs:
        for json_file in posts_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            username = data.get("username", "")
            if not username or username not in voice_map:
                continue
            vt = voice_map[username]["voice_type"]
            if vt == "unknown":
                continue
            seen: set[str] = set()
            for post in data.get("posts", []):
                sub = post.get("subreddit", "")
                if sub:
                    seen.add(sub)
            for comment in data.get("comments", []):
                sub = comment.get("subreddit", "")
                if sub:
                    seen.add(sub)
            for sub in seen:
                sub_voice[sub][vt] += 1
                user_subs[username].add(sub)
    return dict(sub_voice), dict(user_subs)


def _count_shared_users(user_subs: dict[str, set[str]]) -> Counter:
    """Count classified users shared between each subreddit pair — becomes edge weights."""
    edge_counter: Counter = Counter()
    for subs in user_subs.values():
        sub_list = sorted(subs)
        for i in range(len(sub_list)):
            for j in range(i + 1, len(sub_list)):
                edge_counter[(sub_list[i], sub_list[j])] += 1
    return edge_counter


def build_subreddit_graph(
    sub_voice: dict[str, Counter],
    user_subs: dict[str, set[str]],
    min_weight: int,
    min_users: int,
) -> nx.Graph:
    """Build weighted undirected subreddit graph; edge weight = shared classified users."""
    edge_counter = _count_shared_users(user_subs)
    g = nx.Graph()
    for sub, vc in sub_voice.items():
        if sum(vc.values()) < min_users:
            continue
        dominant = vc.most_common(1)[0][0]
        g.add_node(
            sub,
            dominant_voice_type=dominant,
            voice_distribution=json.dumps(dict(vc)),
            user_count=sum(vc.values()),
        )
    for (a, b), weight in edge_counter.items():
        if weight < min_weight or a not in g or b not in g:
            continue
        g.add_edge(a, b, weight=weight)
    return g


# --- community detection ---


def detect_communities(
    graph: nx.Graph, resolution: float = 1.0, random_state: int = 42
) -> dict[str, int]:
    """Run Louvain on the subreddit graph; return subreddit → community_id mapping."""
    return community_louvain.best_partition(
        graph, weight="weight", resolution=resolution, random_state=random_state
    )


# --- output ---


def _generate_rgb_colors(n: int) -> list[dict]:
    """Generate n perceptually distinct RGB dicts for GEXF viz tags."""
    colors = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        colors.append({"r": int(r * 255), "g": int(g * 255), "b": int(b * 255), "a": 1.0})
    return colors


def save_nodes_csv(graph: nx.Graph, partition: dict[str, int], path: Path) -> None:
    """Write per-subreddit attributes — dominant voice type, community, centrality — as CSV."""
    try:
        centrality = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.degree_centrality(graph)
    rows = []
    for node in graph.nodes():
        meta = graph.nodes[node]
        rows.append({
            "subreddit": node,
            "dominant_voice_type": meta.get("dominant_voice_type", "unknown"),
            "voice_distribution": meta.get("voice_distribution", "{}"),
            "user_count": meta.get("user_count", 0),
            "community": partition.get(node, -1),
            "weighted_degree": graph.degree(node, weight="weight"),
            "eigenvector_centrality": round(centrality.get(node, 0.0), 6),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} subreddits → {path}")


def save_gexf(graph: nx.Graph, partition: dict[str, int], path: Path) -> None:
    """Export to GEXF with viz color/size tags for immediate nice appearance in Gephi."""
    print("Computing visual attributes (colors/sizes) for Gephi...")
    voice_types = sorted(
        set(graph.nodes[n].get("dominant_voice_type", "unknown") for n in graph.nodes())
    )
    vt_color = dict(zip(voice_types, _generate_rgb_colors(len(voice_types))))
    max_uc = max((graph.nodes[n].get("user_count", 1) for n in graph.nodes()), default=1)

    for node in graph.nodes():
        meta = graph.nodes[node]
        vt = meta.get("dominant_voice_type", "unknown")
        uc = meta.get("user_count", 1)
        meta["viz"] = {
            "color": vt_color.get(vt, {"r": 128, "g": 128, "b": 128, "a": 1.0}),
            "size": 10.0 + 50.0 * (uc / max(max_uc, 1)),
        }

    nx.write_gexf(graph, path)
    print(f"Saved GEXF ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges) → {path}")


# --- main ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Reddit subreddit co-participation graph (GEXF format for Gephi)"
    )
    p.add_argument(
        "--classifications", nargs="+",
        default=[
            "results/final/user_voice_classification_jan_jun.csv",
            "results/final/user_voice_classification_jul_dec.csv",
        ],
    )
    p.add_argument(
        "--posts-dir", nargs="+",
        default=["results/user_posts_jan_jun", "results/user_posts_jul_dec"],
    )
    p.add_argument("--output-dir", default="results/final")
    p.add_argument(
        "--min-weight", type=int, default=2,
        help="Min shared classified users for an edge to be included",
    )
    p.add_argument(
        "--min-users", type=int, default=2,
        help="Min classified users in a subreddit for it to be included as a node",
    )
    p.add_argument(
        "--resolution", type=float, default=1.0,
        help="Louvain resolution — higher = more smaller communities",
    )
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
    print(f"  {len(voice_map)} classified users")

    print("Building subreddit participation data...")
    sub_voice, user_subs = build_subreddit_data(posts_dirs, voice_map)
    print(f"  {len(sub_voice)} subreddits, {len(user_subs)} users with activity")

    print("Building subreddit graph...")
    graph = build_subreddit_graph(sub_voice, user_subs, args.min_weight, args.min_users)
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    print(f"  Removed {len(isolated)} isolated subreddits ({graph.number_of_nodes()} remaining)")

    print("Detecting communities...")
    partition = detect_communities(graph, resolution=args.resolution)

    singleton_cids = {cid for cid, cnt in Counter(partition.values()).items() if cnt == 1}
    singleton_nodes = [n for n, cid in partition.items() if cid in singleton_cids]
    graph.remove_nodes_from(singleton_nodes)
    partition = {n: cid for n, cid in partition.items() if n in graph}
    print(f"  Removed {len(singleton_nodes)} singleton-community nodes ({graph.number_of_nodes()} remaining)")

    for node, cid in partition.items():
        graph.nodes[node]["community"] = cid

    print("Writing outputs...")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nodes_csv(graph, partition, out_dir / "subreddit_graph_nodes.csv")
    save_gexf(graph, partition, out_dir / "subreddit_graph.gexf")
    print("Done.")


if __name__ == "__main__":
    main()
