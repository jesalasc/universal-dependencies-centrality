from __future__ import annotations

import html
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

from graph_centrality_store import (
    CENTRALITY_METHODS,
    CLOSING_PUNCTUATION,
    DATASET_DEFINITIONS,
    DATABASE_PATH,
    OPENING_PUNCTUATION,
    REPO_ROOT,
    compute_centrality,
    create_connection,
    fetch_centrality_scores,
    get_dataset_graph_dir,
    get_dataset_label,
    get_graph_record,
    list_graph_files,
    load_graph,
    ordered_node_ids,
    remove_punctuation_nodes,
    store_centrality_scores,
    upsert_graph,
)


st.set_page_config(layout="wide")


HISTOGRAM_METRICS = [
    ("node_count", "Tamaño (nodos)", "Número de nodos"),
    ("average_degree", "Grado promedio", "Grado promedio"),
    ("average_out_degree", "Grado de salida promedio", "Grado de salida promedio"),
    ("maximum_degree", "Grado máximo", "Grado máximo"),
    ("diameter", "Diámetro", "Diámetro"),
    ("average_distance", "Distancia promedio", "Distancia promedio"),
]

DATABASE_URL_SECRET = "graph_centralities_db_url"
DATASET_ARCHIVE_URLS_SECRET = "dataset_archive_urls"
AWS_SECTION_SECRET = "aws"
AWS_BUCKET_SECRET = "aws_s3_bucket"
AWS_REGION_SECRET = "aws_region"
AWS_DB_KEY_SECRET = "aws_db_key"
AWS_DATASET_KEYS_SECRET = "aws_dataset_keys"


def get_secret_or_env(name: str, default=None):
    if hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    return os.environ.get(name, default)


def get_section_secret(section_name: str, key: str, default=None):
    if hasattr(st, "secrets") and section_name in st.secrets:
        section = st.secrets[section_name]
        if key in section:
            return section[key]
    return default


def get_aws_setting(key: str, env_name: str, default=None):
    section_value = get_section_secret(AWS_SECTION_SECRET, key)
    if section_value is not None:
        return section_value
    return get_secret_or_env(env_name, default)


def get_dataset_archive_url(dataset_id: str) -> str | None:
    if hasattr(st, "secrets") and DATASET_ARCHIVE_URLS_SECRET in st.secrets:
        archive_urls = st.secrets[DATASET_ARCHIVE_URLS_SECRET]
        if dataset_id in archive_urls:
            return str(archive_urls[dataset_id])

    env_name = f"DATASET_ARCHIVE_URL_{dataset_id.upper()}"
    return os.environ.get(env_name)


def get_dataset_archive_s3_key(dataset_id: str) -> str | None:
    section_keys = get_section_secret(AWS_SECTION_SECRET, "dataset_keys")
    if section_keys and dataset_id in section_keys:
        return str(section_keys[dataset_id])

    if hasattr(st, "secrets") and AWS_DATASET_KEYS_SECRET in st.secrets:
        dataset_keys = st.secrets[AWS_DATASET_KEYS_SECRET]
        if dataset_id in dataset_keys:
            return str(dataset_keys[dataset_id])

    env_name = f"AWS_DATASET_KEY_{dataset_id.upper()}"
    return os.environ.get(env_name)


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "streamlit-centrality-app/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


@st.cache_resource
def get_s3_client():
    session_kwargs = {}
    aws_access_key_id = get_aws_setting("access_key_id", "AWS_ACCESS_KEY_ID")
    aws_secret_access_key = get_aws_setting("secret_access_key", "AWS_SECRET_ACCESS_KEY")
    aws_session_token = get_aws_setting("session_token", "AWS_SESSION_TOKEN")
    region_name = get_aws_setting("region", "AWS_DEFAULT_REGION") or get_secret_or_env(AWS_REGION_SECRET)

    if aws_access_key_id and aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = str(aws_access_key_id)
        session_kwargs["aws_secret_access_key"] = str(aws_secret_access_key)
    if aws_session_token:
        session_kwargs["aws_session_token"] = str(aws_session_token)
    if region_name:
        session_kwargs["region_name"] = str(region_name)

    session = boto3.session.Session(**session_kwargs)
    return session.client("s3")


def download_s3_file(bucket: str, key: str, destination: Path) -> None:
    get_s3_client().download_file(bucket, key, str(destination))


def normalize_extracted_graph_dir(graph_dir: Path) -> None:
    if any(graph_dir.glob("*.graphml")):
        return

    extracted_graphs = [path for path in graph_dir.rglob("*.graphml") if path.parent != graph_dir]
    for graph_path in extracted_graphs:
        target_path = graph_dir / graph_path.name
        if target_path.exists():
            continue
        shutil.move(str(graph_path), str(target_path))


def extract_archive(archive_path: Path, destination_dir: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    normalize_extracted_graph_dir(destination_dir)


def ensure_database_bootstrap() -> str | None:
    if DATABASE_PATH.exists():
        return None

    database_url = get_secret_or_env(DATABASE_URL_SECRET)
    database_s3_key = get_aws_setting("db_key", "AWS_DB_KEY") or get_secret_or_env(AWS_DB_KEY_SECRET)
    s3_bucket = get_aws_setting("bucket", "AWS_S3_BUCKET") or get_secret_or_env(AWS_BUCKET_SECRET)
    if not database_url and not (s3_bucket and database_s3_key):
        return (
            "No se encontró la base de datos de centralidades precomputadas. "
            f"Configura `{DATABASE_URL_SECRET}` o el origen S3 (`{AWS_BUCKET_SECRET}` y `{AWS_DB_KEY_SECRET}`) "
            "en `st.secrets` o como variables de entorno para descargarla automáticamente."
        )

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        temporary_path = Path(temp_dir) / DATABASE_PATH.name
        if database_url:
            download_file(str(database_url), temporary_path)
            source_message = f"Base de datos descargada desde `{DATABASE_URL_SECRET}`."
        else:
            download_s3_file(str(s3_bucket), str(database_s3_key), temporary_path)
            source_message = "Base de datos descargada automáticamente desde S3."
        shutil.move(str(temporary_path), str(DATABASE_PATH))

    return source_message


def ensure_dataset_bootstrap(dataset_id: str) -> str | None:
    graph_dir = get_dataset_graph_dir(dataset_id)
    if graph_dir.exists() and any(graph_dir.glob("*.graphml")):
        return None

    archive_url = get_dataset_archive_url(dataset_id)
    archive_s3_key = get_dataset_archive_s3_key(dataset_id)
    s3_bucket = get_aws_setting("bucket", "AWS_S3_BUCKET") or get_secret_or_env(AWS_BUCKET_SECRET)
    if not archive_url and not (s3_bucket and archive_s3_key):
        env_name = f"DATASET_ARCHIVE_URL_{dataset_id.upper()}"
        return (
            f"No se encontraron grafos para `{get_dataset_label(dataset_id)}`. "
            f"Configura `{DATASET_ARCHIVE_URLS_SECRET}.{dataset_id}` o el origen S3 para ese dataset "
            f"en `st.secrets`, o `{env_name}` como variable de entorno."
        )

    graph_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        temporary_archive = Path(temp_dir) / "dataset_archive"
        if archive_url:
            download_file(str(archive_url), temporary_archive)
        else:
            download_s3_file(str(s3_bucket), str(archive_s3_key), temporary_archive)
        extract_archive(temporary_archive, graph_dir)

    graph_count = sum(1 for _ in graph_dir.glob("*.graphml"))
    if graph_count == 0:
        raise RuntimeError(f"El archivo descargado para {dataset_id} no contiene grafos GraphML utilizables.")

    return f"Corpus `{get_dataset_label(dataset_id)}` descargado automáticamente ({graph_count} grafos)."


def ensure_runtime_assets(dataset_id: str) -> None:
    bootstrap_messages = []

    try:
        database_message = ensure_database_bootstrap()
        if database_message:
            bootstrap_messages.append(("info", database_message))
    except Exception as error:
        bootstrap_messages.append(("warning", f"No se pudo descargar la base de datos precomputada: {error}"))

    try:
        dataset_message = ensure_dataset_bootstrap(dataset_id)
        if dataset_message:
            bootstrap_messages.append(("info", dataset_message))
    except Exception as error:
        bootstrap_messages.append(("warning", f"No se pudo descargar el corpus `{get_dataset_label(dataset_id)}`: {error}"))

    for level, message in bootstrap_messages:
        if level == "warning":
            st.warning(message)
        else:
            st.info(message)


@st.cache_resource
def get_database_connection():
    return create_connection(DATABASE_PATH)


def get_graph_files(dataset_id: str) -> list[str]:
    connection = get_database_connection()
    graph_dir = get_dataset_graph_dir(dataset_id)
    filesystem_files = sorted(path.name for path in graph_dir.glob("*.graphml")) if graph_dir.exists() else []
    graph_files = list_graph_files(connection, dataset_id)
    return sorted(set(graph_files) | set(filesystem_files))


def get_graph_path(dataset_id: str, file_name: str) -> Path:
    connection = get_database_connection()
    record = get_graph_record(connection, dataset_id, file_name)
    if record is not None:
        return REPO_ROOT / str(record["relative_path"])
    return get_dataset_graph_dir(dataset_id) / file_name


def ensure_graph_record_exists(dataset_id: str, file_name: str, directed_graph: nx.DiGraph, graph_path: Path) -> int:
    connection = get_database_connection()
    record = get_graph_record(connection, dataset_id, file_name)
    if record is not None:
        return int(record["id"])

    graph_id = upsert_graph(
        connection,
        dataset_id=dataset_id,
        file_name=file_name,
        relative_path=str(graph_path.relative_to(REPO_ROOT)),
        phrase=str(directed_graph.graph.get("phrase", "")),
        root_node=str(directed_graph.graph.get("root")) if directed_graph.graph.get("root") is not None else None,
        node_count=directed_graph.number_of_nodes(),
        edge_count=directed_graph.number_of_edges(),
    )
    connection.commit()
    return graph_id


def load_selected_graph(dataset_id: str, file_name: str) -> tuple[int, nx.DiGraph, nx.Graph]:
    graph_path = get_graph_path(dataset_id, file_name)
    directed_graph, undirected_graph = load_graph(graph_path)
    graph_id = ensure_graph_record_exists(dataset_id, file_name, directed_graph, graph_path)
    return graph_id, directed_graph, undirected_graph


def apply_punctuation_filter(graph: nx.DiGraph, include_punctuation: bool) -> tuple[nx.DiGraph, nx.Graph]:
    if include_punctuation:
        return graph.copy(), graph.to_undirected()

    filtered_directed = remove_punctuation_nodes(graph)
    return filtered_directed, filtered_directed.to_undirected()


def diameter_and_connectivity(graph: nx.Graph) -> tuple[int, bool]:
    node_count = graph.number_of_nodes()
    if node_count <= 1:
        return 0, True

    if nx.is_connected(graph):
        return int(nx.diameter(graph)), True

    largest_component_diameter = max(
        (
            int(nx.diameter(graph.subgraph(component_nodes).copy()))
            for component_nodes in nx.connected_components(graph)
        ),
        default=0,
    )
    return largest_component_diameter, False


def average_distance_and_connectivity(graph: nx.Graph) -> tuple[float, bool]:
    node_count = graph.number_of_nodes()
    if node_count <= 1:
        return 0.0, True

    if nx.is_connected(graph):
        return float(nx.average_shortest_path_length(graph)), True

    total_distance = 0.0
    total_pairs = 0
    for component_nodes in nx.connected_components(graph):
        component_graph = graph.subgraph(component_nodes).copy()
        component_size = component_graph.number_of_nodes()
        if component_size <= 1:
            continue

        pair_count = component_size * (component_size - 1) // 2
        total_pairs += pair_count
        total_distance += float(nx.average_shortest_path_length(component_graph)) * pair_count

    if total_pairs == 0:
        return 0.0, False

    return total_distance / total_pairs, False


def compute_graph_distribution_metrics(
    directed_graph: nx.DiGraph,
    undirected_graph: nx.Graph,
) -> dict[str, float | int | bool]:
    node_count = undirected_graph.number_of_nodes()
    degree_values = [degree for _, degree in undirected_graph.degree()]
    out_degree_values = [degree for _, degree in directed_graph.out_degree()]
    average_degree = float(sum(degree_values) / node_count) if node_count else 0.0
    average_out_degree = float(sum(out_degree_values) / node_count) if node_count else 0.0
    maximum_degree = int(max(degree_values, default=0))
    diameter, is_connected = diameter_and_connectivity(undirected_graph)
    average_distance, _ = average_distance_and_connectivity(undirected_graph)
    return {
        "node_count": int(node_count),
        "average_degree": average_degree,
        "average_out_degree": average_out_degree,
        "maximum_degree": maximum_degree,
        "diameter": int(diameter),
        "average_distance": average_distance,
        "is_connected": is_connected,
    }


@st.cache_data(show_spinner=False)
def get_dataset_distribution_dataframe(dataset_id: str, graph_files: tuple[str, ...]) -> pd.DataFrame:
    graph_dir = get_dataset_graph_dir(dataset_id)
    rows: list[dict[str, object]] = []

    for file_name in graph_files:
        graph_path = graph_dir / file_name
        if not graph_path.exists():
            graph_path = get_graph_path(dataset_id, file_name)

        directed_graph, undirected_graph = load_graph(graph_path)
        graph_metrics = compute_graph_distribution_metrics(directed_graph, undirected_graph)
        rows.append({"file_name": file_name, **graph_metrics})

    return pd.DataFrame(rows)


def histogram_bins(values: pd.Series) -> np.ndarray | int:
    clean_values = values.dropna()
    if clean_values.empty:
        return 1

    numeric_values = clean_values.to_numpy(dtype=float)
    integer_like = np.allclose(numeric_values, np.round(numeric_values))
    min_value = float(numeric_values.min())
    max_value = float(numeric_values.max())

    if integer_like and (max_value - min_value) <= 30:
        return np.arange(min_value - 0.5, max_value + 1.5, 1.0)

    return min(40, max(10, int(np.sqrt(len(numeric_values)))))


def build_distribution_histograms_figure(dataframe: pd.DataFrame, dataset_label: str):
    columns = 2
    rows = (len(HISTOGRAM_METRICS) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4.5 * rows))
    flat_axes = np.atleast_1d(axes).flatten()

    for axis, (column_name, title, x_label) in zip(flat_axes, HISTOGRAM_METRICS):
        axis.hist(
            dataframe[column_name],
            bins=histogram_bins(dataframe[column_name]),
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Cantidad de grafos")
        axis.grid(axis="y", alpha=0.25)

    for axis in flat_axes[len(HISTOGRAM_METRICS):]:
        axis.remove()

    fig.suptitle(f"Distribuciones estructurales del corpus: {dataset_label}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def render_corpus_histogram_view(dataset_id: str) -> None:
    st.title("Histogramas estructurales del corpus")
    st.caption(f"Dataset activo: {get_dataset_label(dataset_id)}")
    st.caption(
        "Las métricas se calculan por grafo: tamaño, grado promedio, grado máximo, diámetro y distancia promedio en la versión no dirigida; grado de salida promedio en la versión dirigida."
    )

    graph_files = get_graph_files(dataset_id)
    if not graph_files:
        st.error(f"No se encontraron grafos en {get_dataset_graph_dir(dataset_id)}.")
        return

    with st.spinner("Calculando métricas estructurales del corpus..."):
        distribution_dataframe = get_dataset_distribution_dataframe(dataset_id, tuple(graph_files))

    if distribution_dataframe.empty:
        st.warning("No se pudieron calcular métricas para este corpus.")
        return

    metric_columns = [column_name for column_name, _, _ in HISTOGRAM_METRICS]
    summary_dataframe = (
        distribution_dataframe[metric_columns]
        .agg(["min", "mean", "median", "max", "std"])
        .transpose()
        .rename(
            columns={
                "min": "Mínimo",
                "mean": "Media",
                "median": "Mediana",
                "max": "Máximo",
                "std": "Desviación estándar",
            }
        )
    )
    summary_dataframe.index = [
        "Tamaño (nodos)",
        "Grado promedio",
        "Grado de salida promedio",
        "Grado máximo",
        "Diámetro",
        "Distancia promedio",
    ]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Grafos analizados", f"{len(distribution_dataframe):,}")
    with col2:
        st.metric("Tamaño medio", f"{distribution_dataframe['node_count'].mean():.2f}")
    with col3:
        st.metric("Grado promedio medio", f"{distribution_dataframe['average_degree'].mean():.2f}")
    with col4:
        st.metric("Diámetro medio", f"{distribution_dataframe['diameter'].mean():.2f}")

    figure = build_distribution_histograms_figure(distribution_dataframe, get_dataset_label(dataset_id))
    st.pyplot(figure)
    plt.close(figure)

    disconnected_graph_count = int((~distribution_dataframe["is_connected"]).sum())
    if disconnected_graph_count:
        st.info(
            "Se detectaron "
            f"{disconnected_graph_count} grafos no conexos. "
            "Para ellos, el diámetro mostrado corresponde al mayor diámetro entre sus componentes conexas, "
            "y la distancia promedio se calcula como el promedio ponderado por pares alcanzables dentro de cada componente conexa."
        )

    st.markdown("## Resumen estadístico")
    st.dataframe(summary_dataframe.style.format("{:.2f}"))

    with st.expander("Ver métricas por grafo"):
        st.dataframe(
            distribution_dataframe.rename(
                columns={
                    "file_name": "Archivo",
                    "node_count": "Tamaño (nodos)",
                    "average_degree": "Grado promedio",
                    "average_out_degree": "Grado de salida promedio",
                    "maximum_degree": "Grado máximo",
                    "diameter": "Diámetro",
                    "average_distance": "Distancia promedio",
                    "is_connected": "Es conexo",
                }
            ),
            use_container_width=True,
        )


def get_centrality_scores(
    dataset_id: str,
    file_name: str,
    graph_id: int,
    graph: nx.Graph,
    method: str,
    include_punctuation: bool,
) -> dict[str, float]:
    connection = get_database_connection()
    cached_scores = fetch_centrality_scores(connection, dataset_id, file_name, method, include_punctuation)
    if cached_scores is not None:
        return cached_scores

    try:
        centrality_scores = compute_centrality(graph, method)
    except Exception as error:
        st.warning(f"No se pudo calcular {method}: {error}")
        centrality_scores = {str(node_id): 0.0 for node_id in graph.nodes()}

    store_centrality_scores(connection, graph_id, method, include_punctuation, centrality_scores)
    connection.commit()
    return centrality_scores


def build_color_mapping(graph: nx.Graph, centrality_scores: dict[str, float]) -> dict[str, tuple[float, ...]]:
    node_ids = ordered_node_ids(graph)
    if not node_ids:
        return {}

    values = np.array([float(centrality_scores.get(node_id, 0.0)) for node_id in node_ids], dtype=float)
    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 1.0

    if np.isclose(vmin, vmax):
        bins = np.linspace(vmin - 1.0, vmax + 1.0, 11)
    else:
        bins = np.quantile(values, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.linspace(vmin, vmax, 11)

    norm = BoundaryNorm(bins, ncolors=cm.viridis.N, clip=True)
    return {
        node_id: cm.viridis(norm(float(centrality_scores.get(node_id, 0.0))))
        for node_id in node_ids
    }


def draw_graph(
    graph: nx.Graph,
    centrality_scores: dict[str, float],
    title: str,
    positions: dict[str, tuple[float, float]],
    graph_type: str = "undirected",
):
    num_nodes = len(graph.nodes())
    fig_height = max(6, min(0.5 * max(num_nodes, 1), 20))
    fig, ax = plt.subplots(figsize=(8, fig_height))

    if num_nodes == 0:
        ax.text(0.5, 0.5, "Sin nodos para visualizar", ha="center", va="center")
        ax.set_title(title, pad=2)
        ax.axis("off")
        return fig, ax

    node_colors = build_color_mapping(graph, centrality_scores)
    values = np.array(
        [float(centrality_scores.get(str(node_id), 0.0)) for node_id in graph.nodes()],
        dtype=float,
    )
    vmin = float(values.min())
    vmax = float(values.max())

    if np.isclose(vmin, vmax):
        bins = np.linspace(vmin - 1.0, vmax + 1.0, 11)
    else:
        bins = np.quantile(values, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.linspace(vmin, vmax, 11)

    norm = BoundaryNorm(bins, ncolors=cm.viridis.N, clip=True)

    for node_id, (x_pos, y_pos) in positions.items():
        ax.scatter(
            x_pos,
            y_pos,
            s=500,
            color=node_colors.get(str(node_id), cm.viridis(0)),
            edgecolors="black",
            zorder=2,
        )

    nx.draw_networkx_edges(graph, positions, ax=ax)

    if graph_type == "undirected":
        sorted_nodes = sorted(graph.nodes(), key=lambda node_id: centrality_scores.get(str(node_id), 0.0), reverse=True)
        ranking = {str(node_id): index for index, node_id in enumerate(sorted_nodes, start=1)}
        labels = {
            node_id: f"{ranking.get(str(node_id), '')}\n{graph.nodes[node_id].get('form', node_id)}"
            for node_id in graph.nodes()
        }
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, font_color="red", ax=ax)

        scalar_mappable = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        scalar_mappable.set_array([])
        fig.colorbar(scalar_mappable, ax=ax, shrink=0.9, label="Centrality Score")
    else:
        labels = {node_id: graph.nodes[node_id].get("form", node_id) for node_id in graph.nodes()}
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, font_color="red", ax=ax)

    ax.set_title(title, pad=2)
    ax.axis("off")
    plt.subplots_adjust(top=0.75)
    return fig, ax


def hierarchy_pos(graph: nx.DiGraph, root: str | None = None, vert_gap: float = 2):
    if graph.number_of_nodes() == 0:
        return {}

    if root is None or root not in graph:
        roots = [node_id for node_id in graph.nodes if isinstance(graph, nx.DiGraph) and graph.in_degree(node_id) == 0]
        root = roots[0] if roots else next(iter(graph.nodes()))

    def _hierarchy_pos(node_id: str, depth: int = 0, positions: dict | None = None, level_widths: dict | None = None):
        if positions is None:
            positions = {}
        if level_widths is None:
            level_widths = {}

        if depth not in level_widths:
            level_widths[depth] = 0
        else:
            level_widths[depth] += 1

        positions[node_id] = (level_widths[depth], -depth * vert_gap)

        if isinstance(graph, nx.DiGraph):
            children = list(graph.successors(node_id))
        else:
            children = list(graph.neighbors(node_id))

        for child_id in children:
            if child_id not in positions:
                _hierarchy_pos(child_id, depth + 1, positions, level_widths)

        return positions

    positions = _hierarchy_pos(root)
    levels: dict[float, list[float]] = {}

    for node_id, (x_pos, y_pos) in positions.items():
        levels.setdefault(y_pos, []).append(x_pos)

    for y_pos, x_positions in levels.items():
        sorted_positions = sorted(x_positions)
        if len(sorted_positions) == 1:
            mapped_positions = [0.5]
        else:
            mapped_positions = np.linspace(0, 1, len(sorted_positions))

        for index, x_pos in enumerate(sorted_positions):
            for node_id, (current_x, current_y) in positions.items():
                if current_y == y_pos and current_x == x_pos:
                    positions[node_id] = (mapped_positions[index], current_y)

    return positions


def build_dag_from_root(graph: nx.Graph, root_node: str) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_node(root_node, **graph.nodes[root_node])

    visited = {root_node}
    queue = [root_node]

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                dag.add_node(neighbor, **graph.nodes[neighbor])
                dag.add_edge(current_node, neighbor, **graph.edges[current_node, neighbor])
                visited.add(neighbor)
                queue.append(neighbor)

    return dag


def phrase_html(graph: nx.DiGraph, centrality_scores: dict[str, float]) -> str:
    node_colors = build_color_mapping(graph, centrality_scores)
    rendered_phrase = ""
    previous_token = ""

    for node_id in ordered_node_ids(graph):
        token = str(graph.nodes[node_id].get("form", node_id))
        rgba = node_colors.get(node_id, cm.viridis(0))
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
        )
        token_html = f'<span style="color:{hex_color}; font-weight:bold">{html.escape(token)}</span>'

        if not rendered_phrase:
            rendered_phrase = token_html
        elif token in CLOSING_PUNCTUATION:
            rendered_phrase += token_html
        elif previous_token in OPENING_PUNCTUATION:
            rendered_phrase += token_html
        else:
            rendered_phrase += f" {token_html}"

        previous_token = token

    return "**Frase**: " + rendered_phrase


def root_for_display(graph: nx.DiGraph) -> str | None:
    root_node = graph.graph.get("root")
    if root_node is not None and str(root_node) in graph:
        return str(root_node)
    node_ids = ordered_node_ids(graph)
    return node_ids[0] if node_ids else None


def render_single_graph_view(dataset_id: str, file_name: str, include_punctuation: bool, centrality_name: str) -> None:
    graph_id, directed_graph, _ = load_selected_graph(dataset_id, file_name)
    directed_view, undirected_view = apply_punctuation_filter(directed_graph, include_punctuation)
    centrality_scores = get_centrality_scores(
        dataset_id,
        file_name,
        graph_id,
        undirected_view,
        centrality_name,
        include_punctuation,
    )

    st.sidebar.markdown(phrase_html(directed_view, centrality_scores), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Árbol de dependencias (dirigido)")
        root_node = root_for_display(directed_view)
        positions = hierarchy_pos(directed_view, root=root_node, vert_gap=3)
        fig, _ = draw_graph(
            directed_view,
            {str(node_id): 1.0 for node_id in directed_view.nodes()},
            "Árbol sintáctico (dirigido)",
            positions,
            "directed",
        )
        st.pyplot(fig)

    with col2:
        st.subheader(f"Centralidad: {centrality_name}")
        if centrality_scores:
            root_node = max(centrality_scores, key=centrality_scores.get)
            centrality_dag = build_dag_from_root(undirected_view, root_node)
            positions = hierarchy_pos(centrality_dag, root=root_node, vert_gap=3)
            fig, _ = draw_graph(undirected_view, centrality_scores, f"Centralidad: {centrality_name}", positions)
            st.pyplot(fig)
        else:
            st.warning("No se pudo calcular centralidad para este grafo.")


def render_graph_panel(title: str, dataset_id: str, file_name: str, include_punctuation: bool, centrality_name: str) -> None:
    graph_id, directed_graph, _ = load_selected_graph(dataset_id, file_name)
    directed_view, undirected_view = apply_punctuation_filter(directed_graph, include_punctuation)
    centrality_scores = get_centrality_scores(
        dataset_id,
        file_name,
        graph_id,
        undirected_view,
        centrality_name,
        include_punctuation,
    )

    st.subheader(f"{title} — {file_name}")

    with st.expander(f"Frase coloreada ({title})"):
        st.markdown(phrase_html(directed_view, centrality_scores), unsafe_allow_html=True)

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("**Árbol de dependencias (dirigido)**")
        root_node = root_for_display(directed_view)
        positions = hierarchy_pos(directed_view, root=root_node, vert_gap=3)
        fig, _ = draw_graph(
            directed_view,
            {str(node_id): 1.0 for node_id in directed_view.nodes()},
            "Árbol sintáctico (dirigido)",
            positions,
            "directed",
        )
        st.pyplot(fig)

    with right_column:
        st.markdown(f"**Centralidad: {centrality_name}**")
        if centrality_scores:
            root_node = max(centrality_scores, key=centrality_scores.get)
            centrality_dag = build_dag_from_root(undirected_view, root_node)
            positions = hierarchy_pos(centrality_dag, root=root_node, vert_gap=3)
            fig, _ = draw_graph(undirected_view, centrality_scores, f"Centralidad: {centrality_name}", positions)
            st.pyplot(fig)
        else:
            st.warning(f"No se pudo calcular centralidad para {title}.")


dataset_ids = list(DATASET_DEFINITIONS.keys())
selected_dataset = st.sidebar.selectbox(
    "Selecciona el dataset",
    dataset_ids,
    format_func=get_dataset_label,
)

ensure_runtime_assets(selected_dataset)

view = st.sidebar.radio(
    "Selecciona la vista",
    [
        "Visualización de árboles",
        "Visualización múltiple de grafos",
        "Histogramas estructurales del corpus",
        "Datos de distribución",
    ],
)

if view == "Visualización de árboles":
    st.title("Visualizador de centralidad en árboles de dependencias sintácticas")
    st.caption(f"Dataset activo: {get_dataset_label(selected_dataset)}")

    graph_files = get_graph_files(selected_dataset)

    if not graph_files:
        st.error(f"No se encontraron grafos en {get_dataset_graph_dir(selected_dataset)}.")
    else:
        selected_file_name = st.sidebar.selectbox("Selecciona el grafo", graph_files)
        include_punctuation = st.sidebar.checkbox("Incluir puntuación", value=True)
        centrality_name = st.sidebar.selectbox("Medida de centralidad", CENTRALITY_METHODS)
        render_single_graph_view(selected_dataset, selected_file_name, include_punctuation, centrality_name)

elif view == "Visualización múltiple de grafos":
    st.title("Visualización múltiple de grafos")
    st.caption(f"Dataset activo: {get_dataset_label(selected_dataset)}")

    graph_files = get_graph_files(selected_dataset)

    if not graph_files:
        st.error(f"No se encontraron grafos en {get_dataset_graph_dir(selected_dataset)}.")
    else:
        st.sidebar.markdown("### Selección de archivos")
        default_index_a = 0
        default_index_b = 1 if len(graph_files) > 1 else 0

        file_a = st.sidebar.selectbox("Archivo A", graph_files, index=default_index_a, key="file_a_select")
        file_b = st.sidebar.selectbox("Archivo B", graph_files, index=default_index_b, key="file_b_select")

        with st.sidebar.expander("Ajustes del Grafo A", expanded=True):
            include_punctuation_a = st.checkbox("Incluir puntuación (A)", value=True, key="punct_a")
            centrality_a = st.selectbox("Medida de centralidad (A)", CENTRALITY_METHODS, key="cent_a")

        with st.sidebar.expander("Ajustes del Grafo B", expanded=True):
            include_punctuation_b = st.checkbox("Incluir puntuación (B)", value=True, key="punct_b")
            centrality_b = st.selectbox("Medida de centralidad (B)", CENTRALITY_METHODS, key="cent_b")

        if file_a == file_b:
            st.info("Has seleccionado el mismo archivo para A y B. Aún así se muestran ambos paneles.")

        col_a, col_b = st.columns(2)

        with col_a:
            render_graph_panel("Grafo A", selected_dataset, file_a, include_punctuation_a, centrality_a)

        with col_b:
            render_graph_panel("Grafo B", selected_dataset, file_b, include_punctuation_b, centrality_b)

elif view == "Histogramas estructurales del corpus":
    render_corpus_histogram_view(selected_dataset)

elif view == "Datos de distribución":
    st.title("Visualización de distribución")

    csv_path = REPO_ROOT / "distances_to_root.csv"
    if not csv_path.exists():
        st.error(f"No se encontró el archivo {csv_path}.")
    else:
        dataframe = pd.read_csv(csv_path)
        st.markdown("## Vista de las diferencias entre raíces")
        st.dataframe(dataframe)

        st.markdown("## Estadísticas del CSV")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Máximo por columna:**")
            st.dataframe(dataframe.max(numeric_only=True))

        with col2:
            st.write("**Media por columna:**")
            st.dataframe(dataframe.mean(numeric_only=True))

        with col3:
            st.write("**Desviación estándar por columna:**")
            st.dataframe(dataframe.std(numeric_only=True))
