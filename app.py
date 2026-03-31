from __future__ import annotations

import html
from pathlib import Path

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


@st.cache_resource
def get_database_connection():
    return create_connection(DATABASE_PATH)


def get_graph_files(dataset_id: str) -> list[str]:
    connection = get_database_connection()
    graph_files = list_graph_files(connection, dataset_id)
    if graph_files:
        return graph_files

    graph_dir = get_dataset_graph_dir(dataset_id)
    if not graph_dir.exists():
        return []
    return sorted(path.name for path in graph_dir.glob("*.graphml"))


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

view = st.sidebar.radio(
    "Selecciona la vista",
    ["Visualización de árboles", "Visualización múltiple de grafos", "Datos de distribución"],
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
