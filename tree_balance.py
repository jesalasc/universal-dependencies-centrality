from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx

from graph_centrality_store import REPO_ROOT, TREE_BALANCE_INDEXES, ordered_node_ids


TREE_BALANCE_R_SCRIPT = REPO_ROOT / "scripts" / "compute_tree_balance.R"


@dataclass(frozen=True)
class TreeBalancePayload:
    graph_id: int
    include_punctuation: bool
    newick: str | None
    node_count: int
    leaf_count: int
    internal_node_count: int
    suppressed_unary_nodes: int
    synthetic_root_added: bool


def _sorted_successors(graph: nx.DiGraph, node_id: str) -> list[str]:
    child_ids = [str(child_id) for child_id in graph.successors(node_id)]
    ordered_lookup = {node_id: index for index, node_id in enumerate(ordered_node_ids(graph))}
    return sorted(child_ids, key=lambda child_id: ordered_lookup.get(child_id, len(ordered_lookup)))


def _rooted_tree_for_balance(graph: nx.DiGraph) -> tuple[nx.DiGraph, str | None, bool]:
    if graph.number_of_nodes() == 0:
        return graph.copy(), None, False

    rooted_graph = nx.DiGraph()
    rooted_graph.add_nodes_from((str(node_id), attributes) for node_id, attributes in graph.nodes(data=True))
    rooted_graph.add_edges_from((str(source), str(target)) for source, target in graph.edges())

    roots = [str(node_id) for node_id, in_degree in rooted_graph.in_degree() if in_degree == 0]
    preferred_root = graph.graph.get("root")
    if preferred_root is not None and str(preferred_root) in rooted_graph:
        root_node = str(preferred_root)
    elif roots:
        root_node = roots[0]
    else:
        root_node = ordered_node_ids(rooted_graph)[0]

    reachable = nx.descendants(rooted_graph, root_node) | {root_node}
    if len(roots) == 1 and roots[0] == root_node and len(reachable) == rooted_graph.number_of_nodes():
        return rooted_graph, root_node, False

    synthetic_root = "__tree_balance_root__"
    suffix = 0
    while synthetic_root in rooted_graph:
        suffix += 1
        synthetic_root = f"__tree_balance_root_{suffix}__"

    rooted_graph.add_node(synthetic_root)
    attachment_roots = roots or [root_node]
    for attachment_root in attachment_roots:
        if attachment_root != synthetic_root:
            rooted_graph.add_edge(synthetic_root, attachment_root)

    reachable = nx.descendants(rooted_graph, synthetic_root) | {synthetic_root}
    for node_id in ordered_node_ids(rooted_graph):
        if node_id != synthetic_root and node_id not in reachable:
            rooted_graph.add_edge(synthetic_root, node_id)

    return rooted_graph, synthetic_root, True


def build_tree_balance_payload(
    graph_id: int,
    graph: nx.DiGraph,
    include_punctuation: bool,
) -> TreeBalancePayload:
    rooted_graph, root_node, synthetic_root_added = _rooted_tree_for_balance(graph)
    if root_node is None:
        return TreeBalancePayload(
            graph_id=graph_id,
            include_punctuation=include_punctuation,
            newick=None,
            node_count=0,
            leaf_count=0,
            internal_node_count=0,
            suppressed_unary_nodes=0,
            synthetic_root_added=False,
        )

    leaf_count = 0
    internal_node_count = 0
    suppressed_unary_nodes = 0

    def render(node_id: str) -> str:
        nonlocal leaf_count, internal_node_count, suppressed_unary_nodes
        children = _sorted_successors(rooted_graph, node_id)
        if not children:
            leaf_count += 1
            return f"t{leaf_count}"

        rendered_children = [render(child_id) for child_id in children]
        if len(rendered_children) == 1:
            if not str(node_id).startswith("__tree_balance_root__"):
                suppressed_unary_nodes += 1
            return rendered_children[0]

        internal_node_count += 1
        return f"({','.join(rendered_children)})"

    newick = f"{render(root_node)};"
    return TreeBalancePayload(
        graph_id=graph_id,
        include_punctuation=include_punctuation,
        newick=newick,
        node_count=graph.number_of_nodes(),
        leaf_count=leaf_count,
        internal_node_count=internal_node_count,
        suppressed_unary_nodes=suppressed_unary_nodes,
        synthetic_root_added=synthetic_root_added,
    )


def empty_tree_balance_rows(message: str) -> list[dict[str, object]]:
    return [
        {
            "index_name": index["name"],
            "value": None,
            "status": "error",
            "error_message": message,
        }
        for index in TREE_BALANCE_INDEXES
    ]


def compute_tree_balance_batch(payloads: Iterable[TreeBalancePayload]) -> dict[tuple[int, bool], list[dict[str, object]]]:
    payload_list = list(payloads)
    results: dict[tuple[int, bool], list[dict[str, object]]] = {}
    valid_payloads: list[TreeBalancePayload] = []

    for payload in payload_list:
        key = (payload.graph_id, payload.include_punctuation)
        if not payload.newick or payload.leaf_count == 0:
            results[key] = empty_tree_balance_rows("Empty tree after filtering")
            continue
        valid_payloads.append(payload)

    if not valid_payloads:
        return results

    if not TREE_BALANCE_R_SCRIPT.exists():
        raise FileNotFoundError(f"Missing R helper script: {TREE_BALANCE_R_SCRIPT}")

    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary_dir:
        temporary_path = Path(temporary_dir)
        input_path = temporary_path / "tree_balance_input.csv"
        output_path = temporary_path / "tree_balance_output.csv"

        with input_path.open("w", newline="", encoding="utf-8") as input_file:
            writer = csv.DictWriter(input_file, fieldnames=["graph_id", "include_punctuation", "newick"])
            writer.writeheader()
            for payload in valid_payloads:
                writer.writerow(
                    {
                        "graph_id": payload.graph_id,
                        "include_punctuation": int(payload.include_punctuation),
                        "newick": payload.newick,
                    }
                )

        environment = os.environ.copy()
        environment.setdefault("LC_ALL", "C")
        environment.setdefault("LANG", "C")
        command = [
            environment.get("RSCRIPT_BIN", "Rscript"),
            str(TREE_BALANCE_R_SCRIPT),
            str(input_path),
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or "Rscript failed without output"
            raise RuntimeError(
                "Unable to compute tree-balance indices with R. "
                "Install the R packages `treebalance` and `ape`, then rerun the cache builder. "
                f"R output: {detail}"
            )

        with output_path.open(newline="", encoding="utf-8") as output_file:
            reader = csv.DictReader(output_file)
            for row in reader:
                graph_id = int(row["graph_id"])
                include_punctuation = bool(int(row["include_punctuation"]))
                value_text = row.get("value") or ""
                value = float(value_text) if value_text else None
                key = (graph_id, include_punctuation)
                results.setdefault(key, []).append(
                    {
                        "index_name": row["index_name"],
                        "value": value,
                        "status": row.get("status") or "ok",
                        "error_message": row.get("error_message") or None,
                    }
                )

    for payload in payload_list:
        key = (payload.graph_id, payload.include_punctuation)
        if key not in results:
            results[key] = empty_tree_balance_rows("No R result was returned")

    return results
