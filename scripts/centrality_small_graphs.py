"""Compute betweenness and All-Subgraphs centrality for every connected graph
with at most 6 nodes.

For each non-isomorphic connected graph (1..6 nodes) taken from NetworkX's graph
atlas, we emit one CSV row per node containing:
  - graph identifiers (atlas id, #nodes, #edges, edge list)
  - betweenness centrality (raw counts and normalized)
  - All-Subgraphs centrality (asg_cen.all_subgraphs_centrality)

Run from the repository root so that the ``asg_cen`` package is importable:
    python3 scripts/centrality_small_graphs.py
"""

import csv
import os
import sys

import networkx as nx
import numpy as np

# Make sure the repo root is on the path when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asg_cen.all_subgraphs_centrality import all_subgraphs_centrality

MAX_NODES = 6
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_CSV = os.path.join(_ROOT, "centrality_small_graphs.csv")
THRESHOLD_CSV = os.path.join(_ROOT, "betweenness_diversity_by_two_pow_asg.csv")
TREE_KATZ_CSV = os.path.join(_ROOT, "katz_diversity_trees_by_two_pow_asg.csv")
PAGERANK_CSV = os.path.join(_ROOT, "pagerank_values_small_graphs.csv")

# Round ASG-squared values to this many decimals when grouping "equal" values,
# to avoid splitting groups due to floating-point noise.
ASG_DECIMALS = 9


def eigenvector_centrality(g):
    """Eigenvector centrality = Perron (dominant) eigenvector of the adjacency
    matrix, normalized to unit L2 norm with non-negative sign. Computed on the
    dense matrix so it works for any graph size, including n<=2 where the
    ARPACK-based networkx solver fails."""
    nodes = sorted(g.nodes(), key=lambda u: int(u))
    a = nx.to_numpy_array(g, nodelist=nodes)
    vals, vecs = np.linalg.eigh(a)  # symmetric (undirected) -> real spectrum
    vec = vecs[:, int(np.argmax(vals))]
    if vec.sum() < 0:
        vec = -vec
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return {node: float(vec[i]) for i, node in enumerate(nodes)}


def connected_graphs(max_nodes):
    """Yield (atlas_id, graph) for every connected atlas graph with
    1..max_nodes nodes. Atlas graphs are non-isomorphic and use integer
    node labels 0..n-1."""
    for atlas_id, g in enumerate(nx.graph_atlas_g()):
        n = g.number_of_nodes()
        if n == 0 or n > max_nodes:
            continue
        if not nx.is_connected(g):
            continue
        yield atlas_id, g


def main():
    rows = []
    n_graphs = 0
    for atlas_id, g in connected_graphs(MAX_NODES):
        n_graphs += 1
        # Node labels are integers 0..n-1; asg_cen returns string keys.
        edges_str = ";".join(f"{u}-{v}" for u, v in sorted(g.edges()))

        bet_raw = nx.betweenness_centrality(g, normalized=False)
        bet_norm = nx.betweenness_centrality(g, normalized=True)
        pagerank = nx.pagerank(g, alpha=0.8)
        # Katz with default parameters (alpha=0.1, beta=1.0, default nstart).
        katz = nx.katz_centrality(g)
        eigenvector = eigenvector_centrality(g)
        clustering = nx.clustering(g)
        asg = all_subgraphs_centrality(g)

        # A connected graph is a tree iff it has exactly n-1 edges.
        is_tree = g.number_of_edges() == g.number_of_nodes() - 1

        for node in sorted(g.nodes()):
            asg_val = asg[str(node)]
            rows.append(
                {
                    "graph_id": atlas_id,
                    "num_nodes": g.number_of_nodes(),
                    "num_edges": g.number_of_edges(),
                    "is_tree": int(is_tree),
                    "edges": edges_str,
                    "node": node,
                    "betweenness": bet_raw[node],
                    "betweenness_normalized": bet_norm[node],
                    "pagerank_alpha_0.8": pagerank[node],
                    "katz_centrality": katz[node],
                    "eigenvector_centrality": eigenvector[node],
                    "clustering_coefficient": clustering[node],
                    "all_subgraphs_centrality": asg_val,
                    "all_subgraphs_centrality_squared": asg_val ** 2,
                    "two_pow_all_subgraphs_centrality": 2 ** asg_val,
                }
            )

    # Order rows by 2^ASG (then by graph/node for a stable tie-break).
    rows.sort(key=lambda r: (r["two_pow_all_subgraphs_centrality"], r["graph_id"], r["node"]))

    fieldnames = [
        "graph_id",
        "num_nodes",
        "num_edges",
        "is_tree",
        "edges",
        "node",
        "betweenness",
        "betweenness_normalized",
        "pagerank_alpha_0.8",
        "katz_centrality",
        "eigenvector_centrality",
        "clustering_coefficient",
        "all_subgraphs_centrality",
        "all_subgraphs_centrality_squared",
        "two_pow_all_subgraphs_centrality",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {n_graphs} connected graphs (<= {MAX_NODES} nodes).")
    print(f"Wrote {len(rows)} node rows to {OUTPUT_CSV}")

    write_threshold_analysis(rows)
    write_tree_katz_analysis(rows)
    write_pagerank_values(rows)


def write_threshold_analysis(rows):
    """For each distinct 2^ASG value s, count how many DISTINCT betweenness
    centrality values occur among nodes whose 2^ASG is <= s (cumulative).
    Produced for both raw and normalized betweenness."""
    # Distinct 2^ASG thresholds, ascending.
    thresholds = sorted({round(r["two_pow_all_subgraphs_centrality"], ASG_DECIMALS) for r in rows})

    out_rows = []
    for s in thresholds:
        subset = [r for r in rows if round(r["two_pow_all_subgraphs_centrality"], ASG_DECIMALS) <= s]
        distinct_bet = {round(r["betweenness"], ASG_DECIMALS) for r in subset}
        distinct_bet_norm = {round(r["betweenness_normalized"], ASG_DECIMALS) for r in subset}
        distinct_pagerank = {round(r["pagerank_alpha_0.8"], ASG_DECIMALS) for r in subset}
        distinct_katz = {round(r["katz_centrality"], ASG_DECIMALS) for r in subset}
        distinct_eigenvector = {round(r["eigenvector_centrality"], ASG_DECIMALS) for r in subset}
        distinct_clustering = {round(r["clustering_coefficient"], ASG_DECIMALS) for r in subset}
        out_rows.append(
            {
                "two_pow_asg": s,
                "num_nodes_at_most": len(subset),
                "num_distinct_betweenness": len(distinct_bet),
                "num_distinct_betweenness_normalized": len(distinct_bet_norm),
                "num_distinct_pagerank_alpha_0.8": len(distinct_pagerank),
                "num_distinct_katz_centrality": len(distinct_katz),
                "num_distinct_eigenvector_centrality": len(distinct_eigenvector),
                "num_distinct_clustering_coefficient": len(distinct_clustering),
            }
        )

    fieldnames = [
        "two_pow_asg",
        "num_nodes_at_most",
        "num_distinct_betweenness",
        "num_distinct_betweenness_normalized",
        "num_distinct_pagerank_alpha_0.8",
        "num_distinct_katz_centrality",
        "num_distinct_eigenvector_centrality",
        "num_distinct_clustering_coefficient",
    ]
    with open(THRESHOLD_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} threshold rows to {THRESHOLD_CSV}")


def write_tree_katz_analysis(rows):
    """Same cumulative analysis as write_threshold_analysis, but restricted to
    TREES (connected graphs with n-1 edges) and only for the Katz index: for
    each distinct 2^ASG value s, count the distinct Katz values among tree nodes
    whose 2^ASG is <= s."""
    tree_rows = [r for r in rows if r["is_tree"]]
    thresholds = sorted({round(r["two_pow_all_subgraphs_centrality"], ASG_DECIMALS) for r in tree_rows})

    out_rows = []
    for s in thresholds:
        subset = [r for r in tree_rows
                  if round(r["two_pow_all_subgraphs_centrality"], ASG_DECIMALS) <= s]
        distinct_katz = {round(r["katz_centrality"], ASG_DECIMALS) for r in subset}
        out_rows.append(
            {
                "two_pow_asg": s,
                "num_tree_nodes_at_most": len(subset),
                "num_distinct_katz_centrality": len(distinct_katz),
            }
        )

    fieldnames = ["two_pow_asg", "num_tree_nodes_at_most", "num_distinct_katz_centrality"]
    with open(TREE_KATZ_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} tree/Katz threshold rows to {TREE_KATZ_CSV}")


def write_pagerank_values(rows):
    """Dedicated table of PageRank (alpha=0.8) values for every node of every
    graph used in the analysis."""
    fieldnames = ["graph_id", "num_nodes", "num_edges", "edges", "node", "pagerank_alpha_0.8"]
    # Keep a graph-natural order (by id, then node) for this lookup table.
    out_rows = sorted(rows, key=lambda r: (r["graph_id"], r["node"]))
    with open(PAGERANK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} PageRank value rows to {PAGERANK_CSV}")


if __name__ == "__main__":
    main()
