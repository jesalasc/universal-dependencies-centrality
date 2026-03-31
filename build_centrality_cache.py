from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx

from graph_centrality_store import (
    CENTRALITY_METHODS,
    DATABASE_PATH,
    DATASET_DEFINITIONS,
    REPO_ROOT,
    compute_centrality,
    count_cached_centralities,
    create_connection,
    detokenize,
    get_dataset_graph_dir,
    graph_phrase,
    load_graph,
    register_datasets,
    remove_punctuation_nodes,
    store_centrality_scores,
    upsert_graph,
)


ANCORA_SOURCE_DEFAULT = Path("/Users/summa/Documents/Cenia/C. Riveros/data/ancora-dep-2.0/es")
EXPECTED_CACHE_ENTRIES_PER_GRAPH = len(CENTRALITY_METHODS) * 2


def parse_ancora_sentences(source_path: Path):
    sentence_rows = []
    sentence_index = 0

    for raw_line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped_line = raw_line.strip()

        if not stripped_line or stripped_line.startswith("#"):
            if sentence_rows:
                sentence_index += 1
                yield sentence_index, sentence_rows
                sentence_rows = []
            continue

        parts = raw_line.split("\t") if "\t" in raw_line else raw_line.split()
        if len(parts) != 8:
            raise ValueError(f"Unexpected row format in {source_path}: {raw_line!r}")

        token_id, form, lemma, xpos, head, relation, upos, features = parts
        sentence_rows.append(
            {
                "id": token_id,
                "form": form,
                "lemma": lemma,
                "xpos": xpos,
                "head": head,
                "relation": relation,
                "upos": upos,
                "features": features,
            }
        )

    if sentence_rows:
        sentence_index += 1
        yield sentence_index, sentence_rows


def build_ancora_graph(sentence_rows: list[dict[str, str]]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for row in sentence_rows:
        graph.add_node(row["id"], form=row["form"])

    root_node = None
    for row in sentence_rows:
        if row["head"] == "0":
            root_node = row["id"]
            continue
        graph.add_edge(row["head"], row["id"])

    if root_node is None:
        raise ValueError("Sentence without root node")

    tokens = [row["form"] for row in sorted(sentence_rows, key=lambda row: int(row["id"]))]
    graph.graph["root"] = root_node
    graph.graph["phrase"] = detokenize(tokens)
    return graph


def build_ancora_graphml_dataset(source_dir: Path, output_dir: Path, rebuild: bool) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written_graphs = 0
    source_documents = 0

    for source_documents, source_path in enumerate(sorted(source_dir.glob("*.csv")), start=1):
        for sentence_index, sentence_rows in parse_ancora_sentences(source_path):
            graph = build_ancora_graph(sentence_rows)
            file_name = f"{source_path.stem}_s{sentence_index:05d}.graphml"
            output_path = output_dir / file_name
            if rebuild or not output_path.exists():
                nx.write_graphml(graph, output_path)
            written_graphs += 1

        if source_documents % 100 == 0:
            print(f"[ancora] processed {source_documents} source files")

    return source_documents, written_graphs


def sync_graph_metadata(connection, dataset_id: str) -> int:
    graph_dir = get_dataset_graph_dir(dataset_id)
    graph_paths = sorted(graph_dir.glob("*.graphml"))

    for index, graph_path in enumerate(graph_paths, start=1):
        directed_graph = nx.read_graphml(graph_path)
        phrase = directed_graph.graph.get("phrase") or graph_phrase(directed_graph)
        root_node = directed_graph.graph.get("root")
        upsert_graph(
            connection,
            dataset_id=dataset_id,
            file_name=graph_path.name,
            relative_path=str(graph_path.relative_to(REPO_ROOT)),
            phrase=str(phrase),
            root_node=str(root_node) if root_node is not None else None,
            node_count=directed_graph.number_of_nodes(),
            edge_count=directed_graph.number_of_edges(),
            source_document=graph_path.stem.rsplit("_s", 1)[0] if dataset_id == "ancora_dep_2_0_es" else None,
            source_sentence_index=int(graph_path.stem.rsplit("_s", 1)[1]) if dataset_id == "ancora_dep_2_0_es" else None,
        )

        if index % 1000 == 0:
            connection.commit()
            print(f"[metadata] indexed {index}/{len(graph_paths)} graphs for {dataset_id}")

    connection.commit()
    return len(graph_paths)


def graphs_missing_cache(connection, dataset_ids: list[str], force: bool) -> list[tuple[int, str]]:
    placeholders = ", ".join("?" for _ in dataset_ids)
    rows = connection.execute(
        f"""
        SELECT
            graphs.id AS graph_id,
            graphs.relative_path AS relative_path,
            COUNT(cache.method) AS cached_entries
        FROM graphs
        LEFT JOIN centrality_cache AS cache
            ON cache.graph_id = graphs.id
        WHERE graphs.dataset_id IN ({placeholders})
        GROUP BY graphs.id, graphs.relative_path
        ORDER BY graphs.dataset_id, graphs.file_name
        """,
        dataset_ids,
    ).fetchall()

    if force:
        return [(int(row["graph_id"]), str(row["relative_path"])) for row in rows]

    return [
        (int(row["graph_id"]), str(row["relative_path"]))
        for row in rows
        if int(row["cached_entries"]) < EXPECTED_CACHE_ENTRIES_PER_GRAPH
    ]


def compute_graph_cache(relative_path: str) -> list[tuple[str, bool, dict[str, float]]]:
    graph_path = REPO_ROOT / relative_path
    _, undirected_graph = load_graph(graph_path)

    graph_variants = {
        True: undirected_graph,
        False: remove_punctuation_nodes(undirected_graph),
    }

    cache_payload = []
    for include_punctuation, graph in graph_variants.items():
        for method in CENTRALITY_METHODS:
            scores = compute_centrality(graph, method)
            cache_payload.append((method, include_punctuation, scores))

    return cache_payload


def build_cache_entries(connection, dataset_ids: list[str], workers: int, force: bool) -> None:
    tasks = graphs_missing_cache(connection, dataset_ids, force)
    if not tasks:
        print("[cache] all requested graphs are already cached")
        return

    print(f"[cache] computing centralities for {len(tasks)} graphs with {workers} worker(s)")
    started_at = time.time()
    completed = 0

    def consume_executor(executor) -> None:
        nonlocal completed
        future_to_graph = {
            executor.submit(compute_graph_cache, relative_path): (graph_id, relative_path)
            for graph_id, relative_path in tasks
        }

        for future in as_completed(future_to_graph):
            graph_id, relative_path = future_to_graph[future]
            payload = future.result()
            for method, include_punctuation, scores in payload:
                store_centrality_scores(connection, graph_id, method, include_punctuation, scores)

            completed += 1
            if completed % 100 == 0:
                connection.commit()
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed else 0.0
                print(f"[cache] {completed}/{len(tasks)} graphs cached ({rate:.2f} graphs/s)")

    try:
        if workers <= 1:
            with ThreadPoolExecutor(max_workers=1) as executor:
                consume_executor(executor)
            return

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                consume_executor(executor)
        except PermissionError:
            print("[cache] process workers unavailable here, falling back to threads")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                consume_executor(executor)
    finally:
        connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GraphML datasets and centrality cache")
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="SQLite database path",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DEFINITIONS.keys()),
        default=sorted(DATASET_DEFINITIONS.keys()),
        help="Datasets to index and cache",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for centrality computation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached centralities even if entries already exist",
    )
    parser.add_argument(
        "--rebuild-ancora-graphs",
        action="store_true",
        help="Rewrite the AnCora GraphML files even if they already exist",
    )
    parser.add_argument(
        "--ancora-source",
        type=Path,
        default=ANCORA_SOURCE_DEFAULT,
        help="Path to the original AnCora CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if "ancora_dep_2_0_es" in args.datasets:
        source_documents, written_graphs = build_ancora_graphml_dataset(
            args.ancora_source,
            get_dataset_graph_dir("ancora_dep_2_0_es"),
            rebuild=args.rebuild_ancora_graphs,
        )
        print(f"[ancora] ready: {source_documents} source files, {written_graphs} graphs")

    connection = create_connection(args.database)
    register_datasets(connection)

    for dataset_id in args.datasets:
        indexed_graphs = sync_graph_metadata(connection, dataset_id)
        print(f"[metadata] {dataset_id}: {indexed_graphs} graphs indexed")

    build_cache_entries(connection, args.datasets, args.workers, args.force)

    for dataset_id in args.datasets:
        print(f"[summary] {dataset_id}: {count_cached_centralities(connection, dataset_id)} cached entries")

    connection.close()


if __name__ == "__main__":
    main()
