"""Visualize every connected graph with at most 6 nodes, labeling each node
with its 2^ASG value (the all-subgraphs count recovered from the log2-scaled
all_subgraphs_centrality). Graphs are laid out in a grid ordered by number of
nodes (then atlas id).

Run from the repository root:
    python3 scripts/visualize_small_graphs.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asg_cen.all_subgraphs_centrality import all_subgraphs_centrality

MAX_NODES = 6
NCOLS = 8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PNG = os.path.join(_ROOT, "small_graphs_two_pow_asg.png")

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
    """Compact betweenness formatting: integers without a decimal tail."""
    return str(int(x)) if float(x).is_integer() else f"{x:.3g}"


def connected_graphs(max_nodes):
    for atlas_id, g in enumerate(nx.graph_atlas_g()):
        n = g.number_of_nodes()
        if n == 0 or n > max_nodes:
            continue
        if not nx.is_connected(g):
            continue
        yield atlas_id, g


def main():
    graphs = list(connected_graphs(MAX_NODES))
    # Order by number of nodes, then atlas id.
    graphs.sort(key=lambda t: (t[1].number_of_nodes(), t[0]))

    n_graphs = len(graphs)
    nrows = (n_graphs + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 2.4, nrows * 2.4))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for idx, (atlas_id, g) in enumerate(graphs):
        ax = axes[idx]
        n = g.number_of_nodes()

        asg = all_subgraphs_centrality(g)
        bet = nx.betweenness_centrality(g, normalized=False)
        # Each node label shows 2^ASG (integer all-subgraphs count) on top
        # and the raw (un-normalized) betweenness centrality below.
        labels = {
            node: f"{round(2 ** asg[str(node)])}\nB={_fmt(bet[node])}"
            for node in g.nodes()
        }

        if n == 1:
            pos = {list(g.nodes())[0]: (0.0, 0.0)}
        else:
            pos = nx.circular_layout(g)

        color = GROUP_COLORS.get(n, "#dddddd")
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#777777", width=1.2)
        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_color=color, node_size=820,
            edgecolors="#333333", linewidths=1.0,
        )
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7.5, font_weight="bold")

        ax.set_title(f"id {atlas_id} · n={n} · m={g.number_of_edges()}", fontsize=8)
        ax.margins(0.18)

    fig.suptitle(
        "Connected graphs with ≤ 6 nodes — node labels: 2^ASG (all-subgraphs count) "
        "and B = raw betweenness; ordered by #nodes",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight")
    print(f"Rendered {n_graphs} graphs to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
