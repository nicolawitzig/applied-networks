"""
Build a subreddit co-participation graph from Reddit user activity.

Nodes: subreddits with at least --min-users classified users active in them
Edges: A — B weighted by number of classified users active in both subreddits
Node attrs: dominant voice type, voice distribution, classified user count
Output: nodes CSV (voice type + community + centrality), interactive HTML
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
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
            "voice_type": row["voice_type"]
            if pd.notna(row["voice_type"])
            else "unknown",
            "subtype": row["subtype"] if pd.notna(row["subtype"]) else "",
            "confidence": float(row["confidence"])
            if pd.notna(row["confidence"])
            else 0.0,
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


def get_community_voice_distribution(
    graph: nx.Graph, partition: dict[str, int]
) -> dict[int, str]:
    """Aggregate voice type breakdown across all subreddit nodes in each community."""
    community_vc: dict[int, Counter] = defaultdict(Counter)
    for node, cid in partition.items():
        if node not in graph:
            continue
        vc = json.loads(graph.nodes[node].get("voice_distribution", "{}"))
        for vt, count in vc.items():
            community_vc[cid][vt] += count
    result = {}
    for cid in sorted(set(partition.values())):
        vc = community_vc.get(cid, Counter())
        total = sum(vc.values())
        if total == 0:
            result[cid] = "no data"
            continue
        parts = [f"{vt}: {count / total:.0%}" for vt, count in vc.most_common(3)]
        result[cid] = ", ".join(parts)
    return result


# --- color / layout helpers ---


def _generate_colors(n: int) -> list[str]:
    """Generate n perceptually distinct hex colors by spacing hues around the HSV wheel."""
    colors = []
    for i in range(max(n, 1)):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.88)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def _community_order_by_interaction(cids: list[int], meta: nx.Graph) -> list[int]:
    """Greedy nearest-neighbor ordering so heavily-interacting community pairs land at adjacent hues."""
    if len(cids) <= 1:
        return list(cids)
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for u, v, data in meta.edges(data=True):
        w = float(data.get("weight", 1))
        weights[(u, v)] = w
        weights[(v, u)] = w
    total = {c: sum(weights[(c, o)] for o in cids if o != c) for c in cids}
    start = max(
        cids, key=lambda c: total[c]
    )  # most-connected community anchors the chain
    visited = [start]
    remaining = set(cids) - {start}
    while remaining:
        last = visited[-1]
        nxt = max(remaining, key=lambda c: weights[(last, c)])
        visited.append(nxt)
        remaining.remove(nxt)
    return visited


def _build_meta_graph(
    graph: nx.Graph, partition: dict[str, int]
) -> tuple[dict[int, list[str]], nx.Graph]:
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


def _community_colors(
    partition: dict[str, int], graph: nx.Graph | None = None
) -> dict[int, str]:
    """Assign distinct hex colors; adjacent hues go to communities with more mutual interaction."""
    cids = sorted(set(partition.values()))
    if graph is not None:
        _, meta = _build_meta_graph(graph, partition)
        cids = _community_order_by_interaction(cids, meta)
    palette = _generate_colors(len(cids))
    return {cid: palette[i] for i, cid in enumerate(cids)}


def _louvain_layout(
    graph: nx.Graph, partition: dict[str, int]
) -> dict[str, tuple[float, float]]:
    """Kamada-kawai on inter-community meta-graph; nodes packed in a tight ring within each community."""
    communities, meta = _build_meta_graph(graph, partition)
    meta_pos = (
        nx.kamada_kawai_layout(meta, weight="weight", scale=5000)
        if len(communities) > 1
        else {next(iter(communities)): (0.0, 0.0)}
    )
    pos: dict[str, tuple[float, float]] = {}
    for cid, members in communities.items():
        cx, cy = meta_pos[cid]
        if len(members) == 1:
            pos[members[0]] = (float(cx), float(cy))
            continue
        r = max(150.0, 120 * math.sqrt(len(members)))
        for j, node in enumerate(members):
            theta = 2 * math.pi * j / len(members)
            pos[node] = (cx + r * math.cos(theta), cy + r * math.sin(theta))
    return pos


def _community_layout(
    graph: nx.Graph, partition: dict[str, int]
) -> dict[str, tuple[float, float]]:
    """Kamada-kawai on inter-community meta-graph, then kamada-kawai within each community."""
    communities, meta = _build_meta_graph(graph, partition)
    meta_pos = (
        nx.kamada_kawai_layout(meta, weight="weight", scale=5000)
        if len(communities) > 1
        else {next(iter(communities)): (0.0, 0.0)}
    )
    pos: dict[str, tuple[float, float]] = {}
    for cid, members in communities.items():
        cx, cy = meta_pos[cid]
        if len(members) == 1:
            pos[members[0]] = (float(cx), float(cy))
            continue
        local_scale = 800 * math.sqrt(len(members))
        sub = graph.subgraph(members)
        local_pos = nx.kamada_kawai_layout(sub, weight="weight", scale=local_scale)
        for node, (x, y) in local_pos.items():
            pos[node] = (float(cx + x), float(cy + y))
    return pos


# --- HTML generation helpers ---


def _build_hull_script(cid_colors: dict[int, str], cid_conn: dict[str, float]) -> str:
    """JS <script> that draws smooth convex hull outlines per community on the vis.js canvas."""
    colors_json = json.dumps({str(k): v for k, v in cid_colors.items()})
    conn_json = json.dumps(cid_conn)  # weighted inter-community degree per community id
    return f"""<script>
(function() {{
  const COMM_COLORS = {colors_json};
  const COMM_CONN = {conn_json};  // total inter-community edge weight — drives label font size
  const _connVals = Object.values(COMM_CONN);
  const _minConn = Math.min(..._connVals);
  const _maxConn = Math.max(..._connVals);
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
  function drawHull(ctx, pts, col, cid) {{
    const n=pts.length;
    const mid=i=>[(pts[i][0]+pts[(i+1)%n][0])/2,(pts[i][1]+pts[(i+1)%n][1])/2];
    ctx.beginPath();
    if (n===1) {{ ctx.arc(pts[0][0],pts[0][1],80,0,2*Math.PI); }}
    else {{ const m0=mid(n-1); ctx.moveTo(m0[0],m0[1]);
      for (let i=0;i<n;i++) {{ const m=mid(i); ctx.quadraticCurveTo(pts[i][0],pts[i][1],m[0],m[1]); }} }}
    ctx.closePath();
    ctx.strokeStyle=col+'bb'; ctx.lineWidth=4; ctx.stroke();
    const cx=pts.reduce((s,p)=>s+p[0],0)/pts.length;
    const cy=pts.reduce((s,p)=>s+p[1],0)/pts.length;
    const avgR=pts.length===1?80:pts.reduce((s,p)=>s+Math.hypot(p[0]-cx,p[1]-cy),0)/pts.length;
    const conn=COMM_CONN[String(cid)]||0;
    const t=_maxConn>_minConn?(conn-_minConn)/(_maxConn-_minConn):0.5;
    const fontSize=Math.round(14+t*80);  // 14–94px scaled by inter-community connection weight
    ctx.font='bold '+fontSize+'px monospace'; ctx.fillStyle='#e6edf3';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('C'+cid, cx, cy);
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
      drawHull(ctx, padHull(pts.length>=3?convexHull(pts):pts, 80), col, cid);
    }});
  }}
  var showHulls = true;
  function init() {{
    if (typeof network==='undefined') {{ setTimeout(init,100); return; }}
    network.on('afterDrawing', ctx=>{{ if (showHulls) renderHulls(ctx); }});
  }}
  init();
  window.toggleHulls = function() {{
    showHulls = !showHulls;
    var btn = document.getElementById('hullToggleBtn');
    if (btn) btn.textContent = showHulls ? 'Hide outlines' : 'Show outlines';
    network.redraw();
  }};
}})();
</script>"""


def _build_legend_html(vt_color: dict[str, str]) -> str:
    """Fixed-position HTML legend mapping dominant voice type to color."""
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{color};flex-shrink:0"></div>'
        f"<span>{vt}</span></div>"
        for vt, color in sorted(vt_color.items())
    )
    return (
        '<div style="position:fixed;top:16px;right:16px;background:#161b22;'
        "border:1px solid #30363d;border-radius:8px;padding:12px 16px;"
        'color:#e6edf3;font-family:monospace;font-size:13px;z-index:9999">'
        '<div style="font-weight:bold;margin-bottom:8px">Dominant Voice Type</div>'
        + items
        + "</div>"
    )


def _build_community_legend_html(
    community_voice: dict[int, str], cid_color: dict[int, str]
) -> str:
    """Fixed-position HTML legend showing voice type breakdown per community."""
    items = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;margin:6px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{cid_color.get(cid, "#888")};flex-shrink:0;margin-top:2px"></div>'
        f'<div style="max-width:250px"><b>C{cid}</b>: {desc}</div></div>'
        for cid, desc in sorted(community_voice.items())
    )
    return (
        '<div style="position:fixed;bottom:16px;right:16px;background:#161b22;'
        "border:1px solid #30363d;border-radius:8px;padding:12px 16px;"
        "color:#e6edf3;font-family:monospace;font-size:12px;z-index:9999;"
        'max-height:40vh;overflow-y:auto">'
        '<div style="font-weight:bold;margin-bottom:8px">Community Voice Types</div>'
        + items
        + "</div>"
    )


def _build_physics_button_html() -> str:
    """Toggle button that starts/stops the vis.js physics simulation."""
    return """<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);z-index:9999">
  <button id="physicsBtn"
    onclick="(function(){
      var running = network.physics.options.enabled;
      network.setOptions({physics:{enabled:!running}});
      document.getElementById('physicsBtn').textContent = running ? 'Start simulation' : 'Stop simulation';
    })()"
    style="background:#161b22;border:1px solid #30363d;border-radius:6px;
           padding:6px 18px;color:#e6edf3;font-family:monospace;font-size:13px;
           cursor:pointer">Start simulation</button>
</div>"""


def _build_hull_toggle_html() -> str:
    """Toggle button that shows/hides community outline hulls."""
    return """<div style="position:fixed;top:16px;left:16px;z-index:9999">
  <button id="hullToggleBtn"
    onclick="toggleHulls()"
    style="background:#161b22;border:1px solid #30363d;border-radius:6px;
           padding:6px 18px;color:#e6edf3;font-family:monospace;font-size:13px;
           cursor:pointer">Hide outlines</button>
</div>"""


def _build_color_toggle_html(
    vt_color: dict[str, str], cid_colors: dict[int, str]
) -> str:
    """Script + button that toggles node coloring between dominant voice type and community."""
    vt_json = json.dumps(vt_color)
    cid_json = json.dumps({str(k): v for k, v in cid_colors.items()})
    return f"""<script>
(function() {{
  var VT_COLORS = {vt_json};
  var CID_COLORS = {cid_json};
  var colorByCommunity = false;
  window.toggleNodeColor = function() {{
    colorByCommunity = !colorByCommunity;
    var updates = network.body.data.nodes.get().map(function(n) {{
      return {{id: n.id, color: colorByCommunity
        ? (CID_COLORS[String(n.community)] || '#888888')
        : (VT_COLORS[n.voice_type] || '#888888')}};
    }});
    network.body.data.nodes.update(updates);
    var btn = document.getElementById('colorToggleBtn');
    if (btn) btn.textContent = colorByCommunity ? 'Color by voice type' : 'Color by community';
  }};
}})();
</script>
<div style="position:fixed;top:60px;left:16px;z-index:9999">
  <button id="colorToggleBtn" onclick="toggleNodeColor()"
    style="background:#161b22;border:1px solid #30363d;border-radius:6px;
           padding:6px 18px;color:#e6edf3;font-family:monospace;font-size:13px;
           cursor:pointer">Color by community</button>
</div>"""


# --- output ---


def save_nodes_csv(graph: nx.Graph, partition: dict[str, int], path: Path) -> None:
    """Write per-subreddit attributes — dominant voice type, community, centrality — as CSV."""
    try:
        centrality = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.degree_centrality(graph)
    rows = []
    for node in graph.nodes():
        meta = graph.nodes[node]
        rows.append(
            {
                "subreddit": node,
                "dominant_voice_type": meta.get("dominant_voice_type", "unknown"),
                "voice_distribution": meta.get("voice_distribution", "{}"),
                "user_count": meta.get("user_count", 0),
                "community": partition.get(node, -1),
                "weighted_degree": graph.degree(node, weight="weight"),
                "eigenvector_centrality": round(centrality.get(node, 0.0), 6),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} subreddits → {path}")


def _populate_network(
    net,
    graph: nx.Graph,
    partition: dict[str, int],
    vt_color: dict[str, str],
    user_counts: dict[str, int],
    max_uc: int,
    degrees: dict[str, float],
    max_deg: float,
    pos: dict[str, tuple[float, float]],
    layout: str,
) -> None:
    """Add all nodes and edges to the pyvis Network object."""
    for node in graph.nodes():
        meta = graph.nodes[node]
        vt = meta.get("dominant_voice_type", "unknown")
        cid = partition.get(node, 0)
        uc = user_counts[node]
        deg = degrees[node]
        node_kwargs: dict = dict(
            label=node,
            size=8 + 42 * (uc / max_uc),
            color=vt_color.get(vt, "#888888"),
            font={"size": int(8 + 30 * (deg / max_deg)), "color": "#e6edf3"},
            title=f"<div>{node}<br>Dominant: {vt}<br>Distribution: {meta.get('voice_distribution', '{}')}<br>Community: {cid}<br>Users: {uc}</div>",
            community=cid,
            voice_type=vt,
        )
        if node in pos:
            x, y = pos[node]
            node_kwargs.update(x=x, y=y)
            if layout != "physics":
                node_kwargs["physics"] = False
        net.add_node(node, **node_kwargs)
    for a, b, data in graph.edges(data=True):
        w = data.get("weight", 1)
        net.add_edge(a, b, value=w, title=f"{w} shared users")


def save_pyvis_html(
    graph: nx.Graph,
    partition: dict[str, int],
    community_voice: dict[int, str],
    path: Path,
    min_edge_weight: int,
    layout: str,
) -> None:
    """Generate interactive Pyvis HTML; nodes sized by classified user count, coloured by dominant voice type."""
    from pyvis.network import Network

    voice_types = sorted(
        set(graph.nodes[n].get("dominant_voice_type", "unknown") for n in graph.nodes())
    )
    vt_color = dict(zip(voice_types, _generate_colors(len(voice_types))))
    user_counts = {n: graph.nodes[n].get("user_count", 1) for n in graph.nodes()}
    max_uc = max(user_counts.values(), default=1)
    degrees = dict(graph.degree(weight="weight"))
    max_deg = max(degrees.values(), default=1)

    filtered = graph.edge_subgraph(
        [
            (a, b)
            for a, b, d in graph.edges(data=True)
            if d.get("weight", 1) >= min_edge_weight
        ]
    )

    if layout == "physics":
        pos = _louvain_layout(graph, partition)
    elif layout == "kamada-kawai":
        pos = {
            n: (float(x), float(y))
            for n, (x, y) in nx.kamada_kawai_layout(
                graph, weight="weight", scale=3000
            ).items()
        }
    elif layout == "louvain-kamada":
        pos = _community_layout(graph, partition)
    else:
        pos = _louvain_layout(graph, partition)

    net = Network(
        height="95vh",
        width="100%",
        bgcolor="#0d1117",
        font_color="#e6edf3",
        directed=False,
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
    )
    _populate_network(
        net,
        filtered,
        partition,
        vt_color,
        user_counts,
        max_uc,
        degrees,
        max_deg,
        pos,
        layout,
    )

    physics_opts = '{"physics":{"enabled":false,"solver":"forceAtlas2Based","forceAtlas2Based":{"gravitationalConstant":-20,"centralGravity":0.03,"springLength":100,"springConstant":0.8,"damping":0.4,"avoidOverlap":0.5},"stabilization":{"enabled":false}},"edges":{"smooth":true,"color":{"opacity":0.4}},"interaction":{"hover":true,"tooltipDelay":100}}'
    static_opts = '{"physics":{"enabled":false},"edges":{"smooth":true,"color":{"opacity":0.4}},"interaction":{"hover":true,"tooltipDelay":100}}'
    net.set_options(physics_opts if layout == "physics" else static_opts)
    net.write_html(str(path))

    cid_colors = _community_colors(partition, graph)
    _, meta = _build_meta_graph(graph, partition)
    cid_conn = {  # sum of inter-community edge weights per community — used to scale hull labels
        str(cid): sum(d["weight"] for _, _, d in meta.edges(cid, data=True))
        for cid in meta.nodes()
    }
    physics_btn = _build_physics_button_html() if layout == "physics" else ""
    injection = "\n".join(
        [
            _build_hull_script(cid_colors, cid_conn),
            _build_legend_html(vt_color),
            _build_community_legend_html(community_voice, cid_colors),
            physics_btn,
            _build_hull_toggle_html(),
            _build_color_toggle_html(vt_color, cid_colors),
        ]
    )
    html = path.read_text()
    path.write_text(html.replace("</body>", injection + "\n</body>"))
    print(
        f"Saved Pyvis HTML ({filtered.number_of_nodes()} nodes, {filtered.number_of_edges()} edges shown) → {path}"
    )


# --- main ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Reddit subreddit co-participation graph"
    )
    p.add_argument(
        "--classifications",
        nargs="+",
        default=[
            "results/final/user_voice_classification_jan_jun.csv",
            "results/final/user_voice_classification_jul_dec.csv",
        ],
    )
    p.add_argument(
        "--posts-dir",
        nargs="+",
        default=["results/user_posts_jan_jun", "results/user_posts_jul_dec"],
    )
    p.add_argument("--output-dir", default="results/final")
    p.add_argument(
        "--min-weight",
        type=int,
        default=2,
        help="Min shared classified users for an edge to be included",
    )
    p.add_argument(
        "--min-edge-weight",
        type=int,
        default=2,
        help="Min edge weight shown in HTML",
    )
    p.add_argument(
        "--min-users",
        type=int,
        default=2,
        help="Min classified users in a subreddit for it to be included as a node",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain resolution — higher = more smaller communities",
    )
    p.add_argument(
        "--layout",
        choices=["physics", "kamada-kawai", "louvain-kamada", "louvain"],
        default="physics",
        help="physics=vis.js force simulation seeded from louvain ring positions; others are static",
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
    print(
        f"  Removed {len(isolated)} isolated subreddits ({graph.number_of_nodes()} remaining)"
    )

    print("Detecting communities...")
    partition = detect_communities(graph, resolution=args.resolution)

    singleton_cids = {
        cid for cid, cnt in Counter(partition.values()).items() if cnt == 1
    }
    singleton_nodes = [n for n, cid in partition.items() if cid in singleton_cids]
    graph.remove_nodes_from(singleton_nodes)
    partition = {n: cid for n, cid in partition.items() if n in graph}
    print(
        f"  Removed {len(singleton_nodes)} singleton-community nodes ({graph.number_of_nodes()} remaining)"
    )

    for node, cid in partition.items():
        graph.nodes[node]["community"] = cid

    community_voice = get_community_voice_distribution(graph, partition)
    print("\n=== Voice Type Distribution per Community ===")
    for cid, desc in community_voice.items():
        print(f"  Community {cid:2d}: {desc}")
    print("=============================================\n")

    print("Writing outputs...")
    save_nodes_csv(graph, partition, out_dir / "subreddit_graph_nodes.csv")
    save_pyvis_html(
        graph,
        partition,
        community_voice,
        out_dir / "subreddit_graph.html",
        args.min_edge_weight,
        args.layout,
    )
    print("Done.")


if __name__ == "__main__":
    main()
