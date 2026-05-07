#!/usr/bin/env python3
"""
Semantic clustering of subreddit posts and comments via sentence embeddings + UMAP + HDBSCAN.

Pipeline:
  flat CSV  →  sentence-transformer embeddings  →  UMAP 2D  →  HDBSCAN clusters
           →  TF-IDF cluster labels  →  interactive HTML + annotated CSV

Usage:
  python cluster_content.py --input results/ethz_flat.csv
  python cluster_content.py --input results/ethz_flat.csv --min-cluster-size 8 --filter-type post
  python cluster_content.py --input results/ethz_flat.csv --model paraphrase-multilingual-MiniLM-L12-v2
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import hdbscan
import numpy as np
import plotly.graph_objects as go
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

SKIP_TEXTS = {"[deleted]", "[removed]"}    # Reddit tombstone strings — no signal for clustering
MIN_TEXT_LEN = 10                          # chars; filters out one-word replies and empty selftext


def load_items(path: str, filter_type: str) -> List[dict]:
    """Load flat CSV; strip rows with missing/deleted text; optionally restrict to post or comment type."""
    items: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if filter_type != "all" and row["type"] != filter_type:
                continue
            text = row["text"].strip()
            if not text or text in SKIP_TEXTS or len(text) < MIN_TEXT_LEN:
                continue
            items.append(row)
    assert items, f"No usable items in {path} with filter_type='{filter_type}'"
    return items


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """Download (once) and run a sentence-transformer model; returns (N, D) float32 array."""
    print(f"Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    assert embeddings.shape[0] == len(texts), "Embedding count mismatch"
    return embeddings


def reduce_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Project embeddings to 2D via UMAP; cosine metric suits sentence vectors."""
    assert embeddings.shape[0] >= n_neighbors, (
        f"Too few items ({embeddings.shape[0]}) for n_neighbors={n_neighbors} — lower --umap-neighbors"
    )
    print(f"UMAP: {embeddings.shape[0]} × {embeddings.shape[1]}  →  {embeddings.shape[0]} × 2 ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    coords: np.ndarray = reducer.fit_transform(embeddings)
    assert coords.shape == (embeddings.shape[0], 2)
    return coords


def cluster_hdbscan(coords: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    Run HDBSCAN on 2D UMAP coords; returns integer label array.
    Label -1 = noise (point did not fit any cluster).
    """
    print(f"HDBSCAN clustering  (min_cluster_size={min_cluster_size})...")
    labels: np.ndarray = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  {n_clusters} clusters found,  {n_noise} noise points")
    return labels


def label_clusters(texts: List[str], labels: np.ndarray, top_k: int = 5) -> Dict[int, str]:
    """
    Assign a keyword label to each cluster using TF-IDF centroid terms.
    Noise cluster (-1) always gets the label 'noise'.
    """
    vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    tfidf_matrix = vec.fit_transform(texts)    # (N, vocab)
    vocab: np.ndarray = np.array(vec.get_feature_names_out())

    cluster_names: Dict[int, str] = {-1: "noise"}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        mean_vec = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        top_words = ", ".join(vocab[mean_vec.argsort()[-top_k:][::-1]])
        cluster_names[cid] = f"C{cid}: {top_words}"
    return cluster_names


def save_clustered_csv(
    items: List[dict],
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    path: str,
) -> None:
    """Append cluster_id and cluster_label columns to the original flat CSV rows."""
    fieldnames = list(items[0].keys()) + ["cluster_id", "cluster_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item, lbl in zip(items, labels):
            w.writerow({**item, "cluster_id": int(lbl), "cluster_label": cluster_names[int(lbl)]})
    print(f"Saved {len(items)} rows → {path}")


def _item_hover(item: dict) -> str:
    """Build the HTML tooltip string shown on hover in the scatter plot."""
    preview = item["text"][:150].replace("<", "&lt;").replace(">", "&gt;")
    ellipsis = "…" if len(item["text"]) > 150 else ""
    return (
        f"<b>{item['type']}</b>  u/{item['author']}<br>"
        f"{item['date']}  ·  score {item['score']}  ·  depth {item.get('depth', 0)}<br><br>"
        f"{preview}{ellipsis}"
    )


def _cluster_colors(cluster_ids: List[int]) -> Dict[int, str]:
    """
    Map cluster IDs to distinct HSL colors; noise (-1) is always translucent grey.
    Named clusters share the HSV wheel evenly so adjacent IDs are maximally distinct.
    """
    named = [cid for cid in cluster_ids if cid != -1]
    n = max(len(named), 1)
    colors: Dict[int, str] = {
        cid: f"hsl({int(i * 360 / n)},70%,58%)"
        for i, cid in enumerate(named)
    }
    colors[-1] = "rgba(160,160,160,0.25)"    # noise: de-emphasised
    return colors


def save_html(
    items: List[dict],
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    path: str,
) -> None:
    """Write a self-contained Plotly scatter HTML; one trace per cluster for a clean legend."""
    colors = _cluster_colors(sorted(set(labels.tolist())))
    fig = go.Figure()

    # noise trace last so named clusters dominate visually
    ordered = [cid for cid in sorted(set(labels.tolist())) if cid != -1] + [-1]
    for cid in ordered:
        mask = labels == cid
        xs = coords[mask, 0].tolist()
        ys = coords[mask, 1].tolist()
        hover = [_item_hover(item) for item, m in zip(items, mask) if m]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=cluster_names[cid],
            marker=dict(size=7, color=colors[cid], opacity=0.9, line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    fig.update_layout(
        title=dict(text="r/ethz content clusters  (UMAP + HDBSCAN)", font_size=16),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font_size=11),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        height=900,
        hovermode="closest",
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    print(f"Saved interactive HTML → {path}")


def _print_summary(cluster_names: Dict[int, str], labels: np.ndarray) -> None:
    """Print a cluster size table to stdout."""
    print("\nCluster summary:")
    print(f"  {'label':<55}  items")
    print(f"  {'-'*55}  -----")
    for cid, name in sorted(cluster_names.items(), key=lambda x: x[0]):
        count = int((labels == cid).sum())
        print(f"  {name:<55}  {count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed, cluster, and visualise r/ethz posts and comments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input results/ethz_flat.csv
  %(prog)s --input results/ethz_flat.csv --min-cluster-size 8 --filter-type post
  %(prog)s --input results/ethz_flat.csv --model paraphrase-multilingual-MiniLM-L12-v2
        """,
    )
    parser.add_argument("--input",            required=True,
                        help="Flat CSV from scrape_subreddit_content.py")
    parser.add_argument("--output-dir",       default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--model",            default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model (default: all-MiniLM-L6-v2). "
                             "Use paraphrase-multilingual-MiniLM-L12-v2 for mixed-language content.")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="HDBSCAN min points to form a cluster (default: 10). "
                             "Lower for small datasets, raise to merge small clusters.")
    parser.add_argument("--umap-neighbors",   type=int, default=15,
                        help="UMAP n_neighbors — controls local vs global structure (default: 15)")
    parser.add_argument("--umap-min-dist",    type=float, default=0.05,
                        help="UMAP min_dist — lower = tighter clusters in the plot (default: 0.05)")
    parser.add_argument("--filter-type",      choices=["post", "comment", "all"], default="all",
                        help="Restrict to posts, comments, or both (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0].replace("_flat", "")

    items = load_items(args.input, args.filter_type)
    print(f"Loaded {len(items)} items  (filter: {args.filter_type})")

    texts = [item["text"] for item in items]
    embeddings = embed_texts(texts, args.model)
    coords     = reduce_umap(embeddings, args.umap_neighbors, args.umap_min_dist)
    labels     = cluster_hdbscan(coords, args.min_cluster_size)
    names      = label_clusters(texts, labels)

    _print_summary(names, labels)

    save_clustered_csv(items, labels, names,
                       os.path.join(args.output_dir, f"{stem}_clustered.csv"))
    save_html(items, coords, labels, names,
              os.path.join(args.output_dir, f"{stem}_clusters.html"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
