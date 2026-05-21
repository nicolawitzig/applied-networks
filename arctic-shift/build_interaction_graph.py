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
import math
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


def get_community_subreddits(graph: nx.DiGraph, partition: dict[str, int], top_n: int = 5) -> dict[int, list[str]]:
    """Get the most frequent subreddits for each community based on internal interactions."""
    community_sub_counts = defaultdict(Counter)
    for u, v, data in graph.edges(data=True):
        cu = partition.get(u)
        cv = partition.get(v)
        if cu is not None and cu == cv:
            subs = data.get("subreddits", "").split(",")
            w = data.get("weight", 1)
            for sub in subs:
                if sub:
                    community_sub_counts[cu][sub] += w
                    
    result = {}
    for cid in sorted(set(partition.values())):
        counts = community_sub_counts.get(cid, Counter())
        if not counts:
            result[cid] = []
        else:
            result[cid] = [f"{sub} ({cnt})" for sub, cnt in counts.most_common(top_n)]
    return result


# --- output ---

def _generate_colors(n: int) -> list[str]:
    """Generate n perceptually distinct hex colors by spacing hues around the HSV wheel."""
    colors = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _community_colors(partition: dict[str, int]) -> dict[int, str]:
    """Assign a distinct hex color to each community ID."""
    cids = sorted(set(partition.values()))
    palette = _generate_colors(len(cids))
    return {cid: palette[i] for i, cid in enumerate(cids)}


def _build_hull_script(cid_colors: dict[int, str]) -> str:
    """JS <script> that draws smooth convex hull outlines per community on the vis.js canvas."""
    colors_json = json.dumps({str(k): v for k, v in cid_colors.items()})
    return f"""<script>
(function() {{
  const COMM_COLORS = {colors_json};
  function cross(O,A,B) {{ return (A[0]-O[0])*(B[1]-O[1])-(A[1]-O[1])*(B[0]-O[0]); }}
  function convexHull(pts) {{
    if (pts.length < 3) return pts.slice();
    let s = pts.reduce((m,p,i,a) => p[0]<a[m][0] ? i : m, 0);
    const h=[]; let c=s;
    do {{
      h.push(pts[c]);
      let n=(c+1)%pts.length;
      for (let i=0;i<pts.length;i++) if (cross(pts[c],pts[n],pts[i])<0) n=i;
      c=n;
    }} while (c!==s);
    return h;
  }}
  function padHull(hull, pad) {{
    const cx=hull.reduce((s,p)=>s+p[0],0)/hull.length;
    const cy=hull.reduce((s,p)=>s+p[1],0)/hull.length;
    return hull.map(p=>{{ const dx=p[0]-cx,dy=p[1]-cy,l=Math.hypot(dx,dy)||1;
      return [p[0]+dx/l*pad, p[1]+dy/l*pad]; }});
  }}
  function drawHull(ctx, pts, col) {{
    const n=pts.length;
    const mid=i=>[(pts[i][0]+pts[(i+1)%n][0])/2,(pts[i][1]+pts[(i+1)%n][1])/2];
    ctx.beginPath();
    if (n===1) {{ ctx.arc(pts[0][0],pts[0][1],80,0,2*Math.PI); }}
    else {{ const m0=mid(n-1); ctx.moveTo(m0[0],m0[1]);
      for (let i=0;i<n;i++) {{ const m=mid(i); ctx.quadraticCurveTo(pts[i][0],pts[i][1],m[0],m[1]); }} }}
    ctx.closePath();
    ctx.fillStyle=col+'28'; ctx.fill();
    ctx.strokeStyle=col+'bb'; ctx.lineWidth=4; ctx.stroke();
  }}
  function renderHulls(ctx) {{
    const pos=network.getPositions(), groups={{}};
    network.body.data.nodes.get().forEach(n=>{{
      if (!pos[n.id]||n.community==null) return;
      const k=String(n.community);
      (groups[k]=groups[k]||[]).push([pos[n.id].x,pos[n.id].y]);
    }});
    Object.entries(groups).forEach(([cid,pts])=>{{
      const col=COMM_COLORS[cid]||'#ffffff';
      drawHull(ctx, padHull(pts.length>=3?convexHull(pts):pts, 80), col);
    }});
  }}
  function init() {{
    if (typeof network==='undefined') {{ setTimeout(init,100); return; }}
    network.on('afterDrawing', ctx=>renderHulls(ctx));
  }}
  init();
}})();
</script>"""


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


def save_voice_correlation_csv(graph: nx.DiGraph, path: Path) -> None:
    """Calculate and export tally of intra-voice vs extra-voice interactions."""
    stats = defaultdict(lambda: {"count": 0, "weight": 0})
    source_totals = defaultdict(float)
    
    for src, dst, data in graph.edges(data=True):
        src_voice = graph.nodes[src].get("voice_type", "unknown")
        dst_voice = graph.nodes[dst].get("voice_type", "unknown")
        w = data.get("weight", 1)
        stats[(src_voice, dst_voice)]["count"] += 1
        stats[(src_voice, dst_voice)]["weight"] += w
        source_totals[src_voice] += w
        
    rows = []
    for (src_voice, dst_voice), agg in stats.items():
        interaction_type = "intra-voice" if src_voice == dst_voice else "inter-voice"
        fraction = agg["weight"] / source_totals[src_voice] if source_totals[src_voice] > 0 else 0.0
        rows.append({
            "source_voice": src_voice,
            "target_voice": dst_voice,
            "interaction_type": interaction_type,
            "edge_count": agg["count"],
            "total_weight": agg["weight"],
            "fraction_of_source_total": round(fraction, 6),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["source_voice", "total_weight"], ascending=[True, False])
    df.to_csv(path, index=False)
    print(f"Saved voice correlation stats → {path}")


def _group_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Kamada-Kawai on the full graph — cross-group distances drive positioning."""
    raw = nx.kamada_kawai_layout(graph.to_undirected(), weight="weight", scale=3000)
    return {node: (float(x), float(y)) for node, (x, y) in raw.items()}


def _build_meta_graph(graph: nx.DiGraph, partition: dict[str, int]) -> tuple[dict[int, list[str]], nx.Graph]:
    """Build community membership map and inter-community weighted meta-graph."""
    communities: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        if node in graph:
            communities[cid].append(node)
    meta = nx.Graph()
    for cid in communities:
        meta.add_node(cid)
    for src, dst, data in graph.edges(data=True):
        csrc, cdst = partition.get(src), partition.get(dst)
        if csrc is None or cdst is None or csrc == cdst:
            continue
        w = data.get("weight", 1)
        if meta.has_edge(csrc, cdst):
            meta[csrc][cdst]["weight"] += w
        else:
            meta.add_edge(csrc, cdst, weight=w)
    return communities, meta


def _community_layout(graph: nx.DiGraph, partition: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Kamada-kawai on inter-community meta-graph, then kamada-kawai within each community."""
    communities, meta = _build_meta_graph(graph, partition)
    meta_pos = nx.kamada_kawai_layout(meta, weight="weight", scale=5000) if len(communities) > 1 \
        else {next(iter(communities)): (0.0, 0.0)}
    pos: dict[str, tuple[float, float]] = {}
    for cid, members in communities.items():
        cx, cy = meta_pos[cid]
        if len(members) == 1:
            pos[members[0]] = (float(cx), float(cy))
            continue
        local_scale = 800 * math.sqrt(len(members))
        sub = graph.subgraph(members).to_undirected()
        local_pos = nx.kamada_kawai_layout(sub, weight="weight", scale=local_scale)
        for node, (x, y) in local_pos.items():
            pos[node] = (float(cx + x), float(cy + y))
    return pos


def _louvain_layout(graph: nx.DiGraph, partition: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Kamada-kawai on inter-community meta-graph; nodes packed in a tight ring within each community."""
    communities, meta = _build_meta_graph(graph, partition)
    meta_pos = nx.kamada_kawai_layout(meta, weight="weight", scale=5000) if len(communities) > 1 \
        else {next(iter(communities)): (0.0, 0.0)}
    pos: dict[str, tuple[float, float]] = {}
    for cid, members in communities.items():
        cx, cy = meta_pos[cid]
        if len(members) == 1:
            pos[members[0]] = (float(cx), float(cy))
            continue
        r = max(150.0, 120 * math.sqrt(len(members)))  # tight ring — small communities stay compact
        for j, node in enumerate(members):
            theta = 2 * math.pi * j / len(members)
            pos[node] = (cx + r * math.cos(theta), cy + r * math.sin(theta))
    return pos


def _build_legend_html(vt_color: dict[str, str]) -> str:
    """Build a fixed-position HTML legend overlay mapping voice type to color."""
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{color};flex-shrink:0"></div>'
        f'<span>{vt}</span></div>'
        for vt, color in sorted(vt_color.items())
    )
    return (
        '<div style="position:fixed;top:16px;right:16px;background:#161b22;'
        'border:1px solid #30363d;border-radius:8px;padding:12px 16px;'
        'color:#e6edf3;font-family:monospace;font-size:13px;z-index:9999">'
        '<div style="font-weight:bold;margin-bottom:8px">Voice Type</div>'
        + items + "</div>"
    )


def _build_community_legend_html(community_topics: dict[int, list[str]], cid_color: dict[int, str]) -> str:
    """Build a fixed-position HTML legend overlay mapping community ID to top subreddits."""
    items = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;margin:6px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{cid_color.get(cid, "#888")};flex-shrink:0;margin-top:2px"></div>'
        f'<div style="max-width:250px"><b>C{cid}</b>: {", ".join(topics) if topics else "none"}</div></div>'
        for cid, topics in sorted(community_topics.items())
    )
    return (
        '<div style="position:fixed;bottom:16px;right:16px;background:#161b22;'
        'border:1px solid #30363d;border-radius:8px;padding:12px 16px;'
        'color:#e6edf3;font-family:monospace;font-size:12px;z-index:9999;'
        'max-height:40vh;overflow-y:auto">'
        '<div style="font-weight:bold;margin-bottom:8px">Community Topics</div>'
        + items + "</div>"
    )


def save_pyvis_html(
    graph: nx.DiGraph,
    partition: dict[str, int],
    path: Path,
    community_topics: dict[int, list[str]],
    min_edge_weight: int = 1,
    layout: str = "physics",
) -> None:
    """Generate interactive directed Pyvis HTML; nodes sized by in-degree, coloured by voice type."""
    from pyvis.network import Network

    voice_types = sorted(set(graph.nodes[n].get("voice_type", "unknown") for n in graph.nodes()))
    vt_color = dict(zip(voice_types, _generate_colors(len(voice_types))))  # voice_type → hex color

    total_degs = {n: graph.in_degree(n, weight="weight") + graph.out_degree(n, weight="weight") for n in graph.nodes()}
    max_deg = max(total_degs.values(), default=1)
    sorted_degs = sorted(total_degs.values())
    label_threshold = sorted_degs[len(sorted_degs) // 3]  # bottom third gets no label

    max_in = max((graph.in_degree(n, weight="weight") for n in graph.nodes()), default=1)
    if layout == "physics":
        pos = None
    elif layout == "kamada-kawai":
        pos = _group_layout(graph)
    elif layout == "louvain-kamada":
        pos = _community_layout(graph, partition)
    else:  # louvain
        pos = _louvain_layout(graph, partition)

    net = Network(
        height="95vh", width="100%",
        bgcolor="#0d1117", font_color="#e6edf3",
        directed=True, select_menu=True, filter_menu=True,
        cdn_resources="in_line",
    )

    for node in graph.nodes():
        meta = graph.nodes[node]
        vt = meta.get("voice_type", "unknown")
        cid = partition.get(node, 0)
        in_deg = graph.in_degree(node, weight="weight")
        deg = total_degs[node]
        label = node if deg >= label_threshold else ""  # hide label for low-connectivity nodes
        font_size = int(8 + 20 * (deg / max_deg))       # scale font by total degree
        topics_str = ", ".join(community_topics.get(cid, []))
        node_kwargs: dict = dict(
            label=label,
            size=10 + 50 * (in_deg / max(max_in, 1)),
            color=vt_color.get(vt, "#888888"),
            font={"size": font_size, "color": "#e6edf3"},
            title=f"<div>{node}<br>Voice: {vt}<br>Community: {cid}<br>Topics: {topics_str}<br>In-degree: {in_deg}</div>",
            community=cid,  # exposed to JS for hull drawing
            voice_type=vt,  # exposed to JS
        )
        if pos is not None:
            x, y = pos[node]
            node_kwargs.update(x=x, y=y, physics=False)
        net.add_node(node, **node_kwargs)

    n_shown = 0
    for src, dst, data in graph.edges(data=True):
        w = data.get("weight", 1)
        if w >= min_edge_weight:
            net.add_edge(src, dst, value=w, title=f"{w} interactions")
            n_shown += 1

    if layout == "physics":
        net.set_options('{"physics":{"enabled":true,"solver":"forceAtlas2Based","forceAtlas2Based":{"gravitationalConstant":-50,"centralGravity":0.01,"springLength":100,"springConstant":0.08,"damping":0.4,"avoidOverlap":0.5},"stabilization":{"enabled":true,"iterations":1000,"updateInterval":25}},"edges":{"smooth":true,"color":{"opacity":0.4}},"interaction":{"hover":true,"tooltipDelay":100}}')
    else:
        net.set_options('{"physics":{"enabled":false},"edges":{"smooth":true,"color":{"opacity":0.4}},"interaction":{"hover":true,"tooltipDelay":100}}')
    net.write_html(str(path))

    cid_colors = _community_colors(partition)
    html = path.read_text()  # inject hull script and legend overlay
    injection = _build_hull_script(cid_colors) + "\n" + _build_legend_html(vt_color) + "\n" + _build_community_legend_html(community_topics, cid_colors)
    path.write_text(html.replace("</body>", injection + "\n</body>"))
    print(f"Saved Pyvis HTML ({graph.number_of_nodes()} nodes, {n_shown} edges shown) → {path}")


# --- main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Reddit voice interaction graph")
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
    p.add_argument("--min-edge-weight", type=int, default=1, help="Min edge weight shown in HTML")
    p.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution — higher = more smaller communities")
    p.add_argument("--include-unknown", action="store_true", default=False, help="Include unknown-voice users that interact with at least one classified user")
    p.add_argument("--layout", choices=["physics", "kamada-kawai", "louvain-kamada", "louvain"], default="physics", help="physics=vis.js force simulation; kamada-kawai=full-graph static; louvain-kamada=meta-graph kamada-kawai between communities + kamada-kawai within; louvain=meta-graph kamada-kawai between communities + tight ring within")
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

    community_topics = get_community_subreddits(graph, partition, top_n=3)
    print("\n=== Most Common Subreddits per Community (Internal Interactions) ===")
    for cid, topics in community_topics.items():
        print(f"  Community {cid:2d}: {', '.join(topics) if topics else '(no internal subreddits)'}")
    print("==================================================================\n")

    print("Writing outputs...")
    save_nodes_csv(graph, partition, out_dir / "interaction_graph_nodes.csv")
    save_voice_correlation_csv(graph, out_dir / "voice_interaction_correlation.csv")
    save_pyvis_html(graph, partition, out_dir / "interaction_graph.html", community_topics, args.min_edge_weight, layout=args.layout)
    print("Done.")


if __name__ == "__main__":
    main()
