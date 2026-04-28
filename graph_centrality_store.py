from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import networkx as nx

from asg_cen.all_subgraphs_centrality import all_subgraphs_centrality as asg


REPO_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = REPO_ROOT / "graph_centralities.sqlite"

CENTRALITY_METHODS = [
    "Betweenness",
    "Closeness",
    "Harmonic",
    "All-Subgraphs",
    "PageRank",
]

TREE_BALANCE_INDEXES = [
    {"name": "sackin", "label": "Sackin"},
    {"name": "average_leaf_depth", "label": "Profundidad media de hojas"},
    {"name": "variance_leaf_depth", "label": "Varianza de profundidad de hojas"},
    {"name": "total_cophenetic", "label": "Cophenetic total"},
    {"name": "area_per_pair", "label": "Área por par"},
    {"name": "b1", "label": "B1"},
    {"name": "b2", "label": "B2"},
    {"name": "cherry", "label": "Cherry"},
    {"name": "colless_like_exp_mdm", "label": "Colless-like (exp, MDM)"},
    {"name": "maximum_depth", "label": "Profundidad máxima"},
    {"name": "maximum_width", "label": "Anchura máxima"},
    {"name": "maximum_width_over_depth", "label": "Anchura máxima / profundidad"},
    {"name": "modified_maximum_width_difference", "label": "Diferencia modificada máxima de anchuras"},
    {"name": "rooted_quartet", "label": "Cuartetos enraizados"},
    {"name": "s_shape", "label": "s-shape"},
    {"name": "total_internal_path_length", "label": "Longitud interna total"},
    {"name": "total_path_length", "label": "Longitud total de caminos"},
    {"name": "average_vertex_depth", "label": "Profundidad media de vértices"},
]

TREE_BALANCE_INDEX_LABELS = {index["name"]: index["label"] for index in TREE_BALANCE_INDEXES}

DATASET_DEFINITIONS = {
    "ud_spanish_gsd": {
        "label": "UD Spanish GSD",
        "graph_dir": REPO_ROOT / "UD_Spanish-GSD",
    },
    "ud_spanish_ancora": {
        "label": "UD Spanish AnCora",
        "graph_dir": REPO_ROOT / "UD_Spanish-AnCora",
    },
    "ancora_dep_2_0_es": {
        "label": "AnCora DEP 2.0 (es)",
        "graph_dir": REPO_ROOT / "AnCora-ES",
    },
}

OPENING_PUNCTUATION = {"¿", "¡", "(", "[", "{", "«", '"', "'"}
CLOSING_PUNCTUATION = {".", ",", ";", ":", "?", "!", ")", "]", "}", "»", '"', "'"}
PUNCTUATION_ONLY_RE = re.compile(r"[^\w\s]+$", re.UNICODE)


def create_connection(database_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(database_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    ensure_schema(connection)
    register_datasets(connection)
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            graph_dir TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            phrase TEXT,
            root_node TEXT,
            node_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            source_document TEXT,
            source_sentence_index INTEGER,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
            UNIQUE (dataset_id, file_name)
        );

        CREATE INDEX IF NOT EXISTS idx_graphs_dataset_file
        ON graphs(dataset_id, file_name);

        CREATE TABLE IF NOT EXISTS centrality_cache (
            graph_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            include_punctuation INTEGER NOT NULL,
            scores_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
            PRIMARY KEY (graph_id, method, include_punctuation)
        );

        CREATE TABLE IF NOT EXISTS tree_balance_graph_cache (
            graph_id INTEGER NOT NULL,
            include_punctuation INTEGER NOT NULL,
            node_count INTEGER NOT NULL,
            leaf_count INTEGER NOT NULL,
            internal_node_count INTEGER NOT NULL,
            suppressed_unary_nodes INTEGER NOT NULL,
            synthetic_root_added INTEGER NOT NULL,
            newick TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
            PRIMARY KEY (graph_id, include_punctuation)
        );

        CREATE TABLE IF NOT EXISTS tree_balance_cache (
            graph_id INTEGER NOT NULL,
            include_punctuation INTEGER NOT NULL,
            index_name TEXT NOT NULL,
            value REAL,
            status TEXT NOT NULL DEFAULT 'ok',
            error_message TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE,
            PRIMARY KEY (graph_id, include_punctuation, index_name)
        );

        CREATE INDEX IF NOT EXISTS idx_tree_balance_cache_index
        ON tree_balance_cache(index_name, include_punctuation);
        """
    )


def register_datasets(connection: sqlite3.Connection) -> None:
    for dataset_id, definition in DATASET_DEFINITIONS.items():
        connection.execute(
            """
            INSERT INTO datasets(id, label, graph_dir)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label = excluded.label,
                graph_dir = excluded.graph_dir
            """,
            (dataset_id, definition["label"], str(definition["graph_dir"].relative_to(REPO_ROOT))),
        )
    connection.commit()


def get_dataset_graph_dir(dataset_id: str) -> Path:
    return Path(DATASET_DEFINITIONS[dataset_id]["graph_dir"])


def get_dataset_label(dataset_id: str) -> str:
    return str(DATASET_DEFINITIONS[dataset_id]["label"])


def ordered_node_ids(graph: nx.Graph) -> list[str]:
    def sort_key(node_id: str) -> tuple[int, int | str]:
        text = str(node_id)
        if text.isdigit():
            return (0, int(text))
        return (1, text)

    return sorted((str(node_id) for node_id in graph.nodes()), key=sort_key)


def detokenize(tokens: Iterable[str]) -> str:
    phrase = ""
    previous = ""
    for token in tokens:
        if not token:
            continue
        if not phrase:
            phrase = token
        elif token in CLOSING_PUNCTUATION:
            phrase = f"{phrase}{token}"
        elif previous in OPENING_PUNCTUATION:
            phrase = f"{phrase}{token}"
        else:
            phrase = f"{phrase} {token}"
        previous = token
    return phrase


def graph_phrase(graph: nx.Graph) -> str:
    tokens = [graph.nodes[node_id].get("form", str(node_id)) for node_id in ordered_node_ids(graph)]
    return detokenize(tokens)


def load_graph(path: str | Path) -> tuple[nx.DiGraph, nx.Graph]:
    directed_graph = nx.read_graphml(path)
    directed_graph.graph["phrase"] = directed_graph.graph.get("phrase") or graph_phrase(directed_graph)
    if directed_graph.graph.get("root") is not None:
        directed_graph.graph["root"] = str(directed_graph.graph["root"])
    return directed_graph, directed_graph.to_undirected()


def is_punctuation_form(form: str | None) -> bool:
    normalized = (form or "").strip()
    if not normalized:
        return False
    return PUNCTUATION_ONLY_RE.fullmatch(normalized) is not None


def remove_punctuation_nodes(graph: nx.Graph) -> nx.Graph:
    pruned_graph = graph.copy()
    nodes_to_remove = [
        node_id
        for node_id, attributes in pruned_graph.nodes(data=True)
        if is_punctuation_form(attributes.get("form"))
    ]
    pruned_graph.remove_nodes_from(nodes_to_remove)
    return pruned_graph


def compute_centrality(graph: nx.Graph, method: str) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}

    if method == "Betweenness":
        scores = nx.betweenness_centrality(graph)
    elif method == "PageRank":
        scores = nx.pagerank(graph)
    elif method == "Closeness":
        scores = nx.closeness_centrality(graph)
    elif method == "Harmonic":
        scores = nx.harmonic_centrality(graph)
    elif method == "All-Subgraphs":
        scores = asg(graph)
    else:
        scores = {node_id: 0.0 for node_id in graph.nodes()}

    return {str(node_id): float(value) for node_id, value in scores.items()}


def serialize_scores(scores: dict[str, float]) -> str:
    normalized_scores = {str(node_id): float(value) for node_id, value in scores.items()}
    return json.dumps(normalized_scores, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deserialize_scores(scores_json: str) -> dict[str, float]:
    loaded_scores = json.loads(scores_json)
    return {str(node_id): float(value) for node_id, value in loaded_scores.items()}


def upsert_graph(
    connection: sqlite3.Connection,
    *,
    dataset_id: str,
    file_name: str,
    relative_path: str,
    phrase: str,
    root_node: str | None,
    node_count: int,
    edge_count: int,
    source_document: str | None = None,
    source_sentence_index: int | None = None,
) -> int:
    connection.execute(
        """
        INSERT INTO graphs(
            dataset_id,
            file_name,
            relative_path,
            phrase,
            root_node,
            node_count,
            edge_count,
            source_document,
            source_sentence_index
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, file_name) DO UPDATE SET
            relative_path = excluded.relative_path,
            phrase = excluded.phrase,
            root_node = excluded.root_node,
            node_count = excluded.node_count,
            edge_count = excluded.edge_count,
            source_document = excluded.source_document,
            source_sentence_index = excluded.source_sentence_index
        """,
        (
            dataset_id,
            file_name,
            relative_path,
            phrase,
            root_node,
            node_count,
            edge_count,
            source_document,
            source_sentence_index,
        ),
    )
    row = connection.execute(
        "SELECT id FROM graphs WHERE dataset_id = ? AND file_name = ?",
        (dataset_id, file_name),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Unable to fetch graph metadata for {dataset_id}:{file_name}")
    return int(row["id"])


def list_graph_files(connection: sqlite3.Connection, dataset_id: str) -> list[str]:
    rows = connection.execute(
        "SELECT file_name FROM graphs WHERE dataset_id = ? ORDER BY file_name",
        (dataset_id,),
    ).fetchall()
    return [str(row["file_name"]) for row in rows]


def get_graph_record(connection: sqlite3.Connection, dataset_id: str, file_name: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT id, dataset_id, file_name, relative_path, phrase, root_node, node_count, edge_count
        FROM graphs
        WHERE dataset_id = ? AND file_name = ?
        """,
        (dataset_id, file_name),
    ).fetchone()


def store_centrality_scores(
    connection: sqlite3.Connection,
    graph_id: int,
    method: str,
    include_punctuation: bool,
    scores: dict[str, float],
) -> None:
    connection.execute(
        """
        INSERT INTO centrality_cache(graph_id, method, include_punctuation, scores_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(graph_id, method, include_punctuation) DO UPDATE SET
            scores_json = excluded.scores_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (graph_id, method, int(include_punctuation), serialize_scores(scores)),
    )


def fetch_centrality_scores(
    connection: sqlite3.Connection,
    dataset_id: str,
    file_name: str,
    method: str,
    include_punctuation: bool,
) -> dict[str, float] | None:
    row = connection.execute(
        """
        SELECT cache.scores_json
        FROM centrality_cache AS cache
        JOIN graphs AS graphs ON graphs.id = cache.graph_id
        WHERE graphs.dataset_id = ?
          AND graphs.file_name = ?
          AND cache.method = ?
          AND cache.include_punctuation = ?
        """,
        (dataset_id, file_name, method, int(include_punctuation)),
    ).fetchone()
    if row is None:
        return None
    return deserialize_scores(str(row["scores_json"]))


def count_cached_centralities(connection: sqlite3.Connection, dataset_id: str | None = None) -> int:
    if dataset_id is None:
        row = connection.execute("SELECT COUNT(*) AS total FROM centrality_cache").fetchone()
    else:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM centrality_cache AS cache
            JOIN graphs AS graphs ON graphs.id = cache.graph_id
            WHERE graphs.dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()
    return int(row["total"])


def store_tree_balance_metadata(
    connection: sqlite3.Connection,
    graph_id: int,
    include_punctuation: bool,
    *,
    node_count: int,
    leaf_count: int,
    internal_node_count: int,
    suppressed_unary_nodes: int,
    synthetic_root_added: bool,
    newick: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO tree_balance_graph_cache(
            graph_id,
            include_punctuation,
            node_count,
            leaf_count,
            internal_node_count,
            suppressed_unary_nodes,
            synthetic_root_added,
            newick
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(graph_id, include_punctuation) DO UPDATE SET
            node_count = excluded.node_count,
            leaf_count = excluded.leaf_count,
            internal_node_count = excluded.internal_node_count,
            suppressed_unary_nodes = excluded.suppressed_unary_nodes,
            synthetic_root_added = excluded.synthetic_root_added,
            newick = excluded.newick,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            graph_id,
            int(include_punctuation),
            int(node_count),
            int(leaf_count),
            int(internal_node_count),
            int(suppressed_unary_nodes),
            int(synthetic_root_added),
            newick,
        ),
    )


def store_tree_balance_scores(
    connection: sqlite3.Connection,
    graph_id: int,
    include_punctuation: bool,
    rows: Iterable[dict[str, object]],
) -> None:
    connection.executemany(
        """
        INSERT INTO tree_balance_cache(
            graph_id,
            include_punctuation,
            index_name,
            value,
            status,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(graph_id, include_punctuation, index_name) DO UPDATE SET
            value = excluded.value,
            status = excluded.status,
            error_message = excluded.error_message,
            updated_at = CURRENT_TIMESTAMP
        """,
        [
            (
                graph_id,
                int(include_punctuation),
                str(row["index_name"]),
                row.get("value"),
                str(row.get("status") or "ok"),
                row.get("error_message"),
            )
            for row in rows
        ],
    )


def fetch_tree_balance_scores(
    connection: sqlite3.Connection,
    dataset_id: str,
    file_name: str,
    include_punctuation: bool,
) -> dict[str, float | None] | None:
    rows = connection.execute(
        """
        SELECT cache.index_name, cache.value
        FROM tree_balance_cache AS cache
        JOIN graphs AS graphs ON graphs.id = cache.graph_id
        WHERE graphs.dataset_id = ?
          AND graphs.file_name = ?
          AND cache.include_punctuation = ?
        ORDER BY cache.index_name
        """,
        (dataset_id, file_name, int(include_punctuation)),
    ).fetchall()
    if not rows:
        return None
    return {str(row["index_name"]): row["value"] for row in rows}


def fetch_tree_balance_rows(
    connection: sqlite3.Connection,
    dataset_id: str,
    include_punctuation: bool,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            graphs.file_name,
            graphs.phrase,
            graph_cache.node_count,
            graph_cache.leaf_count,
            graph_cache.internal_node_count,
            graph_cache.suppressed_unary_nodes,
            graph_cache.synthetic_root_added,
            cache.index_name,
            cache.value,
            cache.status,
            cache.error_message
        FROM tree_balance_cache AS cache
        JOIN graphs AS graphs ON graphs.id = cache.graph_id
        LEFT JOIN tree_balance_graph_cache AS graph_cache
            ON graph_cache.graph_id = cache.graph_id
           AND graph_cache.include_punctuation = cache.include_punctuation
        WHERE graphs.dataset_id = ?
          AND cache.include_punctuation = ?
        ORDER BY graphs.file_name, cache.index_name
        """,
        (dataset_id, int(include_punctuation)),
    ).fetchall()


def count_cached_tree_balances(connection: sqlite3.Connection, dataset_id: str | None = None) -> int:
    if dataset_id is None:
        row = connection.execute("SELECT COUNT(*) AS total FROM tree_balance_cache").fetchone()
    else:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM tree_balance_cache AS cache
            JOIN graphs AS graphs ON graphs.id = cache.graph_id
            WHERE graphs.dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()
    return int(row["total"])
