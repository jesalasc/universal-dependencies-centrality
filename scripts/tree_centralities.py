"""Node-level centrality values for trees with at most 6 nodes, rendered as PNGs.

Two families are produced:

FIRST (undirected free trees, 14 of them for n=1..6):
  - Eigenvector centrality per node.
      -> trees_eigenvector_undirected.png

SECOND (directed rooted trees, 37 of them for n=1..6 = non-isomorphic rooted
trees, A000081: 1,1,2,4,9,20). Each rooted tree is oriented BOTH ways:
  - "away from root"  : out-arborescence, edges parent -> child
                        (matches this repo's head -> dependent convention).
  - "toward root"     : in-arborescence, edges child -> parent (reversed).
Centralities:
  - PageRank (alpha=0.8): orientation dependent -> one PNG per orientation.
  - Katz centrality      : orientation dependent -> one PNG per orientation.
  - Eigenvector          : DEGENERATE on a directed tree (nilpotent adjacency
                           matrix), so it is reported as "n/a".
  - Clustering coeff.    : 0 for every node of any tree (no triangles).
    The two degenerate measures share one PNG (orientation independent):
      -> directed_trees_eigenvector_clustering_degenerate.png
  PageRank / Katz:
      -> directed_trees_pagerank_away_from_root.png
      -> directed_trees_pagerank_toward_root.png
      -> directed_trees_katz_away_from_root.png
      -> directed_trees_katz_toward_root.png

Run from the repository root:
    python3 scripts/tree_centralities.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_NODES = 6
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One color per node-count group so size classes are visually distinct.
GROUP_COLORS = {
    1: "#cfe8ff",
    2: "#bfe3c0",
    3: "#ffe0a3",
    4: "#f4b8c1",
    5: "#d9c2f0",
    6: "#bfe6e6",
}
ROOT_EDGE = "#d62728"  # highlight the root node border


def _fmt(x):
    """Compact number formatting: integers without a decimal tail."""
    return str(int(x)) if float(x).is_integer() else f"{x:.3g}"


# ---------------------------------------------------------------------------
# Graph enumeration
# ---------------------------------------------------------------------------

def free_trees(max_nodes):
    """Yield (atlas_id, undirected_tree) for every free tree (connected,
    n-1 edges) with 1..max_nodes nodes. 14 of them for max_nodes=6."""
    for atlas_id, g in enumerate(nx.graph_atlas_g()):
        n = g.number_of_nodes()
        if n == 0 or n > max_nodes:
            continue
        if not nx.is_connected(g):
            continue
        if g.number_of_edges() != n - 1:
            continue
        yield atlas_id, g


def rooted_trees(max_nodes):
    """Return the list of non-isomorphic rooted trees with 1..max_nodes nodes,
    each as an OUT-arborescence (edges directed away from the root).

    Built by rooting every free tree at every node and de-duplicating by
    directed-graph isomorphism (the root is the unique in-degree-0 node, so
    directed isomorphism == rooted-tree isomorphism). Nodes are relabelled in
    BFS order with the root as 0 for a clean, deterministic display.

    Returns a list of dicts sorted by (#nodes, structure): {id, D, root, n}.
    """
    kept_by_n = {}
    for _atlas_id, g in free_trees(max_nodes):
        for root in g.nodes():
            arb = nx.bfs_tree(g, root)  # edges point away from root
            n = arb.number_of_nodes()
            bucket = kept_by_n.setdefault(n, [])
            if any(nx.is_isomorphic(arb, kept) for kept in bucket):
                continue
            # Relabel to BFS order, root -> 0.
            order = list(nx.bfs_tree(arb, root))
            mapping = {old: i for i, old in enumerate(order)}
            canon = nx.relabel_nodes(arb, mapping)
            bucket.append(canon)

    trees = []
    for n in sorted(kept_by_n):
        # Deterministic within-n order: by depth multiset then edge list.
        def signature(d):
            depths = nx.shortest_path_length(d, 0)
            return (tuple(sorted(depths.values())), tuple(sorted(d.edges())))

        for d in sorted(kept_by_n[n], key=signature):
            trees.append({"D": d, "root": 0, "n": n})

    for i, t in enumerate(trees):
        t["id"] = i
    return trees


# ---------------------------------------------------------------------------
# Centrality helpers
# ---------------------------------------------------------------------------

def eigenvector_undirected(g):
    """Eigenvector centrality = dominant (Perron) eigenvector of the adjacency
    matrix, unit L2 norm, non-negative sign. Dense solve so it also works for
    n<=2 where the ARPACK-based networkx solver fails."""
    nodes = sorted(g.nodes(), key=lambda u: int(u))
    a = nx.to_numpy_array(g, nodelist=nodes)
    vals, vecs = np.linalg.eigh(a)
    vec = vecs[:, int(np.argmax(vals))]
    if vec.sum() < 0:
        vec = -vec
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return {node: float(vec[i]) for i, node in enumerate(nodes)}


def katz_directed(D):
    """Katz centrality on a directed graph via the direct (numpy) solve, which
    is robust on the acyclic adjacency matrices of arborescences. Defaults
    match nx.katz_centrality (alpha=0.1, beta=1.0, unit-norm)."""
    return nx.katz_centrality_numpy(D, alpha=0.1, beta=1.0, normalized=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def hierarchy_pos(out_arb, root):
    """Top-down tidy layout for a rooted tree given its out-arborescence.
    Root at the top; leaves spread evenly left-to-right."""
    children = {u: list(out_arb.successors(u)) for u in out_arb.nodes()}
    pos = {}
    cursor = [0.0]

    def place(node, depth):
        kids = children.get(node, [])
        if not kids:
            x = cursor[0]
            cursor[0] += 1.0
        else:
            xs = [place(k, depth + 1) for k in kids]
            x = sum(xs) / len(xs)
        pos[node] = (x, -float(depth))
        return x

    place(root, 0)
    return pos


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _grid(n_items, ncols):
    nrows = (n_items + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.7, nrows * 2.7))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    return fig, axes


def render_undirected_eigenvector(out_png):
    graphs = list(free_trees(MAX_NODES))
    graphs.sort(key=lambda t: (t[1].number_of_nodes(), t[0]))
    fig, axes = _grid(len(graphs), ncols=5)

    for idx, (atlas_id, g) in enumerate(graphs):
        ax = axes[idx]
        n = g.number_of_nodes()
        ev = eigenvector_undirected(g)
        labels = {node: _fmt(ev[node]) for node in g.nodes()}
        pos = {list(g.nodes())[0]: (0.0, 0.0)} if n == 1 else nx.spring_layout(g, seed=42)
        color = GROUP_COLORS.get(n, "#dddddd")
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#777777", width=1.2)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=color, node_size=1000,
                               edgecolors="#333333", linewidths=1.0)
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=8, font_weight="bold")
        ax.set_title(f"id {atlas_id} · n={n}", fontsize=8)
        ax.margins(0.22)

    fig.suptitle(
        "Undirected trees ≤ 6 nodes — node label: eigenvector centrality "
        "(unit-norm Perron vector)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {len(graphs)} undirected trees -> {out_png}")


def render_directed(trees, out_png, value_fn, orientation, title):
    """Render the 37 rooted trees with a per-node value.

    value_fn(drawn_digraph) -> dict node->label_value or None (=> "n/a").
    orientation: "away" (out-arborescence) or "toward" (in-arborescence).
    """
    fig, axes = _grid(len(trees), ncols=6)

    for idx, t in enumerate(trees):
        ax = axes[idx]
        out_arb, root, n = t["D"], t["root"], t["n"]
        drawn = out_arb if orientation == "away" else out_arb.reverse(copy=True)
        pos = hierarchy_pos(out_arb, root)

        vals = value_fn(drawn)
        if vals is None:
            labels = {node: "n/a" for node in out_arb.nodes()}
        else:
            labels = {node: _fmt(vals[node]) for node in out_arb.nodes()}

        color = GROUP_COLORS.get(n, "#dddddd")
        nodes = list(out_arb.nodes())
        node_border = [ROOT_EDGE if u == root else "#333333" for u in nodes]
        node_lw = [2.4 if u == root else 1.0 for u in nodes]

        nx.draw_networkx_edges(
            drawn, pos, ax=ax, edge_color="#777777", width=1.3,
            arrows=True, arrowstyle="-|>", arrowsize=13, node_size=1000,
        )
        nx.draw_networkx_nodes(
            drawn, pos, ax=ax, nodelist=nodes, node_color=color, node_size=1000,
            edgecolors=node_border, linewidths=node_lw,
        )
        nx.draw_networkx_labels(drawn, pos, labels=labels, ax=ax,
                                font_size=7.5, font_weight="bold")
        ax.set_title(f"id {t['id']} · n={n}", fontsize=8)
        ax.margins(0.22)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {len(trees)} rooted trees -> {out_png}")


def render_directed_degenerate(trees, out_png):
    """Eigenvector (n/a) and clustering (0) — both orientation independent."""
    fig, axes = _grid(len(trees), ncols=6)
    for idx, t in enumerate(trees):
        ax = axes[idx]
        out_arb, root, n = t["D"], t["root"], t["n"]
        pos = hierarchy_pos(out_arb, root)
        clustering = nx.clustering(out_arb)  # 0 everywhere
        labels = {node: f"EV: n/a\nC={_fmt(clustering[node])}" for node in out_arb.nodes()}
        color = GROUP_COLORS.get(n, "#dddddd")
        nodes = list(out_arb.nodes())
        node_border = [ROOT_EDGE if u == root else "#333333" for u in nodes]
        node_lw = [2.4 if u == root else 1.0 for u in nodes]
        nx.draw_networkx_edges(out_arb, pos, ax=ax, edge_color="#777777", width=1.3,
                               arrows=True, arrowstyle="-|>", arrowsize=13, node_size=1000)
        nx.draw_networkx_nodes(out_arb, pos, ax=ax, nodelist=nodes, node_color=color,
                               node_size=1000, edgecolors=node_border, linewidths=node_lw)
        nx.draw_networkx_labels(out_arb, pos, labels=labels, ax=ax, font_size=6.8,
                                font_weight="bold")
        ax.set_title(f"id {t['id']} · n={n}", fontsize=8)
        ax.margins(0.22)

    fig.suptitle(
        "Directed rooted trees ≤ 6 nodes — DEGENERATE measures.  "
        "EV: eigenvector centrality is undefined on a directed tree "
        "(nilpotent adjacency matrix).  C: clustering coefficient = 0 for "
        "every tree node (no triangles). Orientation independent.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {len(trees)} rooted trees (degenerate panel) -> {out_png}")


def main():
    # ---- FIRST: undirected eigenvector -------------------------------------
    render_undirected_eigenvector(os.path.join(_ROOT, "trees_eigenvector_undirected.png"))

    # ---- SECOND: directed rooted trees -------------------------------------
    trees = rooted_trees(MAX_NODES)
    per_n = {}
    for t in trees:
        per_n[t["n"]] = per_n.get(t["n"], 0) + 1
    print(f"Enumerated {len(trees)} non-isomorphic rooted trees: {per_n}")
    assert len(trees) == 37, f"expected 37 rooted trees, got {len(trees)}"

    render_directed(
        trees, os.path.join(_ROOT, "directed_trees_pagerank_away_from_root.png"),
        value_fn=lambda D: nx.pagerank(D, alpha=0.8), orientation="away",
        title="Directed rooted trees ≤ 6 nodes — PageRank (alpha=0.8), edges AWAY "
              "from root (head→dependent). Root outlined in red.",
    )
    render_directed(
        trees, os.path.join(_ROOT, "directed_trees_pagerank_toward_root.png"),
        value_fn=lambda D: nx.pagerank(D, alpha=0.8), orientation="toward",
        title="Directed rooted trees ≤ 6 nodes — PageRank (alpha=0.8), edges TOWARD "
              "root (dependent→head). Root outlined in red.",
    )
    render_directed(
        trees, os.path.join(_ROOT, "directed_trees_katz_away_from_root.png"),
        value_fn=katz_directed, orientation="away",
        title="Directed rooted trees ≤ 6 nodes — Katz centrality (alpha=0.1), edges "
              "AWAY from root (head→dependent). Root outlined in red.",
    )
    render_directed(
        trees, os.path.join(_ROOT, "directed_trees_katz_toward_root.png"),
        value_fn=katz_directed, orientation="toward",
        title="Directed rooted trees ≤ 6 nodes — Katz centrality (alpha=0.1), edges "
              "TOWARD root (dependent→head). Root outlined in red.",
    )
    render_directed_degenerate(
        trees, os.path.join(_ROOT, "directed_trees_eigenvector_clustering_degenerate.png"),
    )


if __name__ == "__main__":
    main()
