"""
Step 4 (LLM variant): build the @mention network from LLM-categorized accounts,
run Louvain, write GraphML + summary.

Usage:
    python scripts/04_LLM_build_network.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.network import build_mention_graph, community_summary, detect_communities
from src.utils.io import load_config, read_csv, read_jsonl, resolve_path, write_csv


def main() -> None:
    cfg = load_config()
    raw = resolve_path(cfg["paths"]["raw"])
    categorized = resolve_path(cfg["paths"].get("categorized", "data/categorized"))

    accounts = read_csv(categorized / "accounts_llm.csv")
    tweets = pd.DataFrame(read_jsonl(raw / "tweets.jsonl"))

    kept_handles = set(accounts["handle"].str.lower())

    g = build_mention_graph(
        tweets,
        kept_handles=kept_handles,
        min_edge_weight=cfg["network"]["min_mentions_for_edge"],
    )
    print(f"[04-LLM] graph: {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")

    partition = detect_communities(g, resolution=cfg["network"]["louvain_resolution"])
    for node, cid in partition.items():
        g.nodes[node]["community"] = cid

    graphml_path = categorized / "mention_network_llm.graphml"
    nx.write_graphml(g, graphml_path)
    print(f"[04-LLM] wrote {graphml_path}")

    summary = community_summary(g, partition, accounts)
    write_csv(summary, categorized / "community_summary_llm.csv")
    print(f"[04-LLM] communities: {len(summary)}")
    print(summary)


if __name__ == "__main__":
    main()
