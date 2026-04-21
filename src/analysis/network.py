"""
Mention network + Louvain community detection (RQ3).

Edges: directed, weighted by # of mentions from account A to account B,
restricted to pairs where both endpoints are in our kept-accounts set.
Louvain operates on the undirected version (python-louvain requirement).
"""
from __future__ import annotations

from collections import Counter

import community as community_louvain
import networkx as nx
import pandas as pd


def build_mention_graph(
    tweets: pd.DataFrame,
    kept_handles: set[str],
    min_edge_weight: int = 1,
) -> nx.DiGraph:
    g = nx.DiGraph()
    kept_lower = {h.lower() for h in kept_handles}
    edge_weights: Counter = Counter()

    for _, row in tweets.iterrows():
        src = (row["handle"] or "").lower()
        if src not in kept_lower:
            continue
        for dst in row.get("mentioned_handles", []) or []:
            dst_l = (dst or "").lower()
            if not dst_l or dst_l == src or dst_l not in kept_lower:
                continue
            edge_weights[(src, dst_l)] += 1

    for (src, dst), w in edge_weights.items():
        if w >= min_edge_weight:
            g.add_edge(src, dst, weight=w)
    return g


def detect_communities(graph: nx.DiGraph, resolution: float = 1.0, random_state: int = 42) -> dict[str, int]:
    undirected = graph.to_undirected()
    for u, v, d in graph.edges(data=True):
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] = undirected[u][v].get("weight", 0) + d.get("weight", 1)
    partition = community_louvain.best_partition(
        undirected, weight="weight", resolution=resolution, random_state=random_state
    )
    return partition


def community_summary(
    graph: nx.DiGraph,
    partition: dict[str, int],
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for handle, cid in partition.items():
        rows.append({"handle": handle, "community": cid})
    df = pd.DataFrame(rows).merge(
        accounts[["handle", "voice_type", "subtype", "followers_count"]],
        on="handle", how="left",
    )
    return df.groupby("community").agg(
        size=("handle", "count"),
        top_voices=("voice_type", lambda s: s.value_counts().head(3).to_dict()),
        mean_followers=("followers_count", "mean"),
    ).reset_index()
