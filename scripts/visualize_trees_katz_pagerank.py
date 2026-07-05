"""Visualize every tree with at most 6 nodes, labeling each node with its
Katz centrality (default params) and PageRank (alpha=0.8). Trees are laid out
in a grid ordered by number of nodes (then atlas id).

Run from the repository root:
    python3 scripts/visualize_trees_katz_pagerank.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_NODES = 6
NCOLS = 6
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PNG = os.path.join(_ROOT, "trees_katz_pagerank.png")

# One color per node-count group so size classes are visually distinct.
GROUP_COLORS = {
    1: "#cfe8ff",
    2: "#bfe3c0",
    3: "#ffe0a3",
    4: "#f4b8c1",
    5: "#d9c2f0",
    6: "#bfe6e6",
}


def _fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.3g}"


def trees(max_nodes):
    """Yield (atlas_id, graph) for every tree (connected, n-1 edges) with
    1..max_nodes nodes."""
    for atlas_id, g in enumerate(nx.graph_atlas_g()):
        n = g.number_of_nodes()
        if n == 0 or n > max_nodes:
            continue
        if not nx.is_connected(g):
            continue
        if g.number_of_edges() != n - 1:
            continue
        yield atlas_id, g


def main():
    graphs = list(trees(MAX_NODES))
    graphs.sort(key=lambda t: (t[1].number_of_nodes(), t[0]))

    n_graphs = len(graphs)
    nrows = (n_graphs + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 2.6, nrows * 2.6))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, (atlas_id, g) in enumerate(graphs):
        ax = axes[idx]
        n = g.number_of_nodes()

        katz = nx.katz_centrality(g)
        pagerank = nx.pagerank(g, alpha=0.8)
        labels = {
            node: f"K={_fmt(katz[node])}\nPR={_fmt(pagerank[node])}"
            for node in g.nodes()
        }

        if n == 1:
            pos = {list(g.nodes())[0]: (0.0, 0.0)}
        else:
            pos = nx.spring_layout(g, seed=42)

        color = GROUP_COLORS.get(n, "#dddddd")
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#777777", width=1.2)
        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_color=color, node_size=900,
            edgecolors="#333333", linewidths=1.0,
        )
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

        ax.set_title(f"id {atlas_id} · n={n}", fontsize=8)
        ax.margins(0.22)

    fig.suptitle(
        "Trees with ≤ 6 nodes — node labels: K = Katz centrality, "
        "PR = PageRank (alpha=0.8); ordered by #nodes",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Rendered {n_graphs} trees to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
