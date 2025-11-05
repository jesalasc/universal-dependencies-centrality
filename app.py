import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
# from spacy import displacy
import streamlit.components.v1 as components
import os
import re

# Ensure the all_subgraphs code directory is on the Python path
from asg_cen.all_subgraphs_centrality import all_subgraphs_centrality as asg

st.set_page_config(layout="wide")

# --- View selector ---
view = st.sidebar.radio(
    "Selecciona la vista",
    ["Visualización de árboles", "Visualización múltiple de grafos", "Datos de distribución"]
)

# --- Load GraphML ---
def load_graph(file):
    G = nx.read_graphml(file)
    G.graph['phrase'] = G.graph.get('phrase', 'No phrase found')
    return G, G.to_undirected()

# --- Centrality Computation ---
def compute_centrality(G, method):
    if method == "Betweenness":
        # Shortest-path betweenness centrality.  [oai_citation:1‡networkx.org](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html?utm_source=chatgpt.com)
        return nx.betweenness_centrality(G)
    elif method == "PageRank":
        # PageRank over graph structure.  [oai_citation:2‡networkx.org](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html?utm_source=chatgpt.com)
        return nx.pagerank(G)
    elif method == "Closeness":
        # Reciprocal of average shortest-path distance.  [oai_citation:3‡networkx.org](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html?utm_source=chatgpt.com)
        return nx.closeness_centrality(G)
    elif method == "Harmonic":
        # Sum of reciprocals of distances.  [oai_citation:4‡networkx.org](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.harmonic_centrality.html?utm_source=chatgpt.com)
        return nx.harmonic_centrality(G)
    elif method == "All-Subgraphs":
        try:
            asg_cen = asg(G)
            return asg_cen
        except Exception as e:
            st.warning(f"ASG centrality failed to solve: {e}")
            return {n: 0 for n in G.nodes()}
    else:
        return {n: 0 for n in G.nodes()}

# --- Draw Graph ---
def draw_graph(G, centrality, title, pos, type='undirected'):
    # Adjust height based on tree size
    num_nodes = len(G.nodes())
    fig_height = max(6, min(0.5 * num_nodes, 20))
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Prepare centrality values as floats
    values = np.array(list(centrality.values()), dtype=float)

    # Create quantile-based bins for more discriminative coloring
    num_bins = 10
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    bins = np.quantile(values, np.linspace(0, 1, num_bins + 1))
    norm = BoundaryNorm(bins, ncolors=cm.viridis.N, clip=True)

    cmap = cm.viridis
    node_colors_dict = {node: cmap(norm(value)) for node, value in centrality.items()}

    # Draw nodes individually
    for node, (x, y) in pos.items():
        ax.scatter(
            x, y,
            s=500,
            color=node_colors_dict.get(node, cmap(0)),
            edgecolors='black',
            zorder=2
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax)

    if type == 'undirected':
        # Ranking labels
        try:
            ranking = {}
            sorted_nodes = sorted(centrality, key=lambda n: centrality[n], reverse=True)
            for idx, n in enumerate(sorted_nodes, 1):
                ranking[n] = idx
        except Exception:
            ranking = {n: "" for n in G.nodes()}

        labels = {n: f"{ranking.get(n, '')}\n{G.nodes[n].get('form', n)}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='red', ax=ax)

        # Standalone colorbar via ScalarMappable.  [oai_citation:5‡matplotlib.org](https://matplotlib.org/stable/users/explain/colors/colorbar_only.html?utm_source=chatgpt.com)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.9, label='Centrality Score')
    else:
        labels = {n: G.nodes[n].get('form', n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='red', ax=ax)

    ax.set_title(title, pad=2)
    ax.axis('off')
    plt.subplots_adjust(top=0.75)
    return fig, ax

def hierarchy_pos(G, root=None, width=1.0, vert_gap=2, vert_loc=0, xcenter=0.5):
    if root is None:
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            raise ValueError("No root found")
        root = roots[0]

    def _hierarchy_pos(node, depth=0, pos=None, level_widths=None):
        if pos is None:
            pos = {}
        if level_widths is None:
            level_widths = {}

        if depth not in level_widths:
            level_widths[depth] = 0
        else:
            level_widths[depth] += 1

        x = level_widths[depth]
        y = -depth * vert_gap
        pos[node] = (x, y)

        children = list(G.successors(node))
        for child in children:
            _hierarchy_pos(child, depth + 1, pos, level_widths)

        return pos

    pos = _hierarchy_pos(root)

    # Normalize x coordinates per level to spread evenly in [0, 1] range
    levels = {}
    for node, (x, y) in pos.items():
        if y not in levels:
            levels[y] = []
        levels[y].append(x)

    for y in levels:
        xs = sorted(levels[y])
        n = len(xs)
        if n == 1:
            mapped = [xcenter]
        else:
            mapped = np.linspace(0, 1, n)
        for i, x in enumerate(xs):
            for node, (nx_, ny_) in pos.items():
                if ny_ == y and nx_ == x:
                    pos[node] = (mapped[i], ny_)

    return pos

def build_dag_from_root(G_undirected, root_node):
    """
    Given an undirected graph and a root node, return a DAG
    with edges directed away from the root based on BFS.
    """
    G_dag = nx.DiGraph()
    G_dag.add_node(root_node, **G_undirected.nodes[root_node])

    visited = set([root_node])
    queue = [root_node]

    while queue:
        current = queue.pop(0)
        for neighbor in G_undirected.neighbors(current):
            if neighbor not in visited:
                G_dag.add_node(neighbor, **G_undirected.nodes[neighbor])
                G_dag.add_edge(current, neighbor, **G_undirected.edges[current, neighbor])
                visited.add(neighbor)
                queue.append(neighbor)

    return G_dag

# --- Helper to color the phrase per-centrality ---
def phrase_html(G_directed, G_undirected, centrality_scores):
    cmap = cm.viridis
    values = np.array(list(centrality_scores.values()), dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    bins = np.quantile(values, np.linspace(0, 1, 11))
    norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
    node_colors_dict = {G_undirected.nodes[node].get("form", "UNKNOWN"): cmap(norm(centrality_scores[node])) for node in G_undirected.nodes()}

    phrase = G_directed.graph.get("phrase", "")
    tokens = re.findall(r"\w+|[.,;:]", phrase)

    final_phrase = []
    for word in tokens:
        rgba = node_colors_dict.get(word, cmap(0))
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )
        final_phrase.append(f'<span style="color:{hex_color}; font-weight:bold">{word}</span>')
    return "**Frase**: " + ' '.join(final_phrase)

# --- View logic ---
import pandas as pd

if view == "Visualización de árboles":
    not_wanted = []
    st.title("Visualizador de centralidad en árboles de dependencias sintácticas")

    GRAPH_DIR = "./UD_Spanish-GSD/"  # Set your desired directory path here
    graph_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]

    if not graph_files:
        st.error(f"No .graphml files found in {GRAPH_DIR}.")
    else:
        selected_file_name = st.sidebar.selectbox("Select graph file", graph_files)
        with open(os.path.join(GRAPH_DIR, selected_file_name), 'r', encoding='utf-8') as f:
            G_directed, G_undirected = load_graph(f)

        include_punctuation = st.sidebar.checkbox("Incluir puntuación", value=True)

        if not include_punctuation:
            not_wanted = [".", ",", ";", ":"]
            punct_nodes = [n for n in G_directed.nodes() if G_directed.nodes[n].get("form") in not_wanted]
            G_directed.remove_nodes_from(punct_nodes)
            G_undirected = G_directed.to_undirected()

        st.sidebar.markdown("### Elige una medida de centralidad")
        centrality_name = st.sidebar.selectbox("Medida de centralidad", ["Betweenness", "Closeness", "Harmonic", "All-Subgraphs", "PageRank"])

        centrality_scores = compute_centrality(G_undirected, centrality_name)

        # Colored phrase in sidebar
        st.sidebar.markdown(phrase_html(G_directed, G_undirected, centrality_scores), unsafe_allow_html=True)

        # Display trees side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Árbol de dependencias (dirigido)")
            root_node = str(G_directed.graph.get('root', None))
            pos1 = hierarchy_pos(G_directed, root=root_node, vert_gap=3)
            fig1, _ = draw_graph(G_directed, {n: 1 for n in G_directed.nodes()}, "Árbol sintáctico (dirigido)", pos1, 'directed')
            st.pyplot(fig1)

        with col2:
            st.subheader(f"Centralidad: {centrality_name}")
            c = centrality_scores
            root_node = max(c, key=c.get)
            G_dag = build_dag_from_root(G_undirected, root_node)
            pos2 = hierarchy_pos(G_dag, root=root_node, vert_gap=3)
            fig2, _ = draw_graph(G_undirected, c, f"Centralidad: {centrality_name}", pos2)
            st.pyplot(fig2)

elif view == "Visualización múltiple de grafos":
    st.title("Visualización múltiple de grafos")

    GRAPH_DIR = "./UD_Spanish-GSD/"
    graph_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]

    if not graph_files:
        st.error(f"No .graphml files found in {GRAPH_DIR}.")
    else:
        st.sidebar.markdown("### Selección de archivos")
        # Pre-select first two different files if available
        default_idx_a = 0
        default_idx_b = 1 if len(graph_files) > 1 else 0

        file_a = st.sidebar.selectbox("Archivo A", graph_files, index=default_idx_a, key="file_a_select")
        file_b = st.sidebar.selectbox("Archivo B", graph_files, index=default_idx_b, key="file_b_select")

        # Settings per-graph
        with st.sidebar.expander("Ajustes del Grafo A", expanded=True):
            include_punct_a = st.checkbox("Incluir puntuación (A)", value=True, key="punct_a")
            centrality_a = st.selectbox("Medida de centralidad (A)", ["Betweenness", "Closeness", "Harmonic", "All-Subgraphs", "PageRank"], key="cent_a")

        with st.sidebar.expander("Ajustes del Grafo B", expanded=True):
            include_punct_b = st.checkbox("Incluir puntuación (B)", value=True, key="punct_b")
            centrality_b = st.selectbox("Medida de centralidad (B)", ["Betweenness", "Closeness", "Harmonic", "All-Subgraphs", "PageRank"], key="cent_b")

        if file_a == file_b:
            st.info("Has seleccionado el mismo archivo para A y B. Aún así se muestran ambos paneles (útil para comparar centralidades o filtros).")

        # Load both graphs
        with open(os.path.join(GRAPH_DIR, file_a), 'r', encoding='utf-8') as fa:
            Gd_a, Gu_a = load_graph(fa)
        with open(os.path.join(GRAPH_DIR, file_b), 'r', encoding='utf-8') as fb:
            Gd_b, Gu_b = load_graph(fb)

        # Apply punctuation filters
        if not include_punct_a:
            not_wanted = [".", ",", ";", ":"]
            punct_nodes_a = [n for n in Gd_a.nodes() if Gd_a.nodes[n].get("form") in not_wanted]
            Gd_a.remove_nodes_from(punct_nodes_a)
            Gu_a = Gd_a.to_undirected()

        if not include_punct_b:
            not_wanted = [".", ",", ";", ":"]
            punct_nodes_b = [n for n in Gd_b.nodes() if Gd_b.nodes[n].get("form") in not_wanted]
            Gd_b.remove_nodes_from(punct_nodes_b)
            Gu_b = Gd_b.to_undirected()

        # Compute centralities
        cen_a = compute_centrality(Gu_a, centrality_a)
        cen_b = compute_centrality(Gu_b, centrality_b)

        # Two main columns: A | B
        colA, colB = st.columns(2)

        # ----- Panel A -----
        with colA:
            st.subheader(f"Grafo A — {file_a}")
            # Colored phrase for A
            with st.expander("Frase coloreada (A)"):
                st.markdown(phrase_html(Gd_a, Gu_a, cen_a), unsafe_allow_html=True)

            # Split: Directed | Centrality
            subA1, subA2 = st.columns(2)

            with subA1:
                st.markdown("**Árbol de dependencias (dirigido)**")
                root_a = str(Gd_a.graph.get('root', None))
                posA1 = hierarchy_pos(Gd_a, root=root_a, vert_gap=3)
                figA1, _ = draw_graph(Gd_a, {n: 1 for n in Gd_a.nodes()}, "Árbol sintáctico (dirigido)", posA1, 'directed')
                st.pyplot(figA1)

            with subA2:
                st.markdown(f"**Centralidad: {centrality_a}**")
                root_a_c = max(cen_a, key=cen_a.get) if len(cen_a) else None
                if root_a_c is not None:
                    Gdag_a = build_dag_from_root(Gu_a, root_a_c)
                    posA2 = hierarchy_pos(Gdag_a, root=root_a_c, vert_gap=3)
                    figA2, _ = draw_graph(Gu_a, cen_a, f"Centralidad: {centrality_a}", posA2)
                    st.pyplot(figA2)
                else:
                    st.warning("No se pudo calcular centralidad para el Grafo A.")

        # ----- Panel B -----
        with colB:
            st.subheader(f"Grafo B — {file_b}")
            # Colored phrase for B
            with st.expander("Frase coloreada (B)"):
                st.markdown(phrase_html(Gd_b, Gu_b, cen_b), unsafe_allow_html=True)

            # Split: Directed | Centrality
            subB1, subB2 = st.columns(2)

            with subB1:
                st.markdown("**Árbol de dependencias (dirigido)**")
                root_b = str(Gd_b.graph.get('root', None))
                posB1 = hierarchy_pos(Gd_b, root=root_b, vert_gap=3)
                figB1, _ = draw_graph(Gd_b, {n: 1 for n in Gd_b.nodes()}, "Árbol sintáctico (dirigido)", posB1, 'directed')
                st.pyplot(figB1)

            with subB2:
                st.markdown(f"**Centralidad: {centrality_b}**")
                root_b_c = max(cen_b, key=cen_b.get) if len(cen_b) else None
                if root_b_c is not None:
                    Gdag_b = build_dag_from_root(Gu_b, root_b_c)
                    posB2 = hierarchy_pos(Gdag_b, root=root_b_c, vert_gap=3)
                    figB2, _ = draw_graph(Gu_b, cen_b, f"Centralidad: {centrality_b}", posB2)
                    st.pyplot(figB2)
                else:
                    st.warning("No se pudo calcular centralidad para el Grafo B.")

elif view == "Datos de distribución":
    st.title("Visualización de distrubición")

    uploaded_csv = "./distances_to_root.csv"

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.markdown("## Vista de las diferencias entre raíces")
        st.dataframe(df)

        st.markdown("## Estadísticas del CSV")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Máximo por columna:**")
            st.dataframe(df.max(numeric_only=True))

        with col2:
            st.write("**Media por columna:**")
            st.dataframe(df.mean(numeric_only=True))

        with col3:
            st.write("**Desviación estándar por columna:**")
            st.dataframe(df.std(numeric_only=True))