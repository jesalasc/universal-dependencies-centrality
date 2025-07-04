import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
#çfrom spacy import displacy
import streamlit.components.v1 as components
import os
import re

# Ensure the all_subgraphs code directory is on the Python path
from asg_cen.all_subgraphs_centrality import all_subgraphs_centrality as asg

st.set_page_config(layout="wide")

# --- View selector ---
view = st.sidebar.radio("Selecciona la vista", ["Visualización de árboles", "Datos de distribución"])

    # Pin phrase at top

# --- Load GraphML ---
def load_graph(file):
    G = nx.read_graphml(file)
    G.graph['phrase'] = G.graph.get('phrase', 'No phrase found')
    return G, G.to_undirected()

# --- Centrality Computation ---
def compute_centrality(G, method):
    if method == "Betweenness":
        return nx.betweenness_centrality(G)
    elif method == "PageRank":
        return nx.pagerank(G)
    elif method == "Closeness":
        return nx.closeness_centrality(G)
    elif method == "Harmonic":
        return nx.harmonic_centrality(G)
    elif method == "All-Subgraphs":
        try:
            asg_cen = asg(G)
            #print({G.nodes[n].get('form', n): asg_cen[n] for n in G.nodes()})  # Debugging output
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
    # If all values identical, expand min/max to avoid zero-width bins
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    bins = np.quantile(values, np.linspace(0, 1, num_bins + 1))
    # BoundaryNorm will map values into discrete color bins
    norm = BoundaryNorm(bins, ncolors=cm.viridis.N, clip=True)

    cmap = cm.viridis
    # Build an explicit color mapping per node
    node_colors_dict = {node: cmap(norm(value))
                        for node, value in centrality.items()}

    # Draw each node individually to avoid relying on node order
    for node, (x, y) in pos.items():
        ax.scatter(x, y,
                   s=500,
                   color=node_colors_dict.get(node, cmap(0)),
                   edgecolors='black',
                   zorder=2)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Draw labels with black font color and larger font size
    # Compute centrality ranking
    
    
    if type == 'undirected':
        try:
        # Sort nodes by centrality descending, assign ranking starting from 1
            ranking = {}
            sorted_nodes = sorted(centrality, key=lambda n: centrality[n], reverse=True)
            for idx, n in enumerate(sorted_nodes, 1):
                ranking[n] = idx
        
        except Exception:
            ranking = {n: "" for n in G.nodes()}
            
        labels = {n: f"{ranking.get(n, '')}\n{G.nodes[n].get('form', n)}" for n in G.nodes()}
            
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='red', ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.9, label='Centrality Score')
        
    else:
        labels = {n: G.nodes[n].get('form', n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='red', ax=ax)

    ax.set_title(title, pad=2)
    ax.axis('off')
    plt.subplots_adjust(top=0.75)  # Reduce top margin even further
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
            for node, (nx, ny) in pos.items():
                if ny == y and nx == x:
                    pos[node] = (mapped[i], ny)

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


# --- View logic ---
import pandas as pd

if view == "Visualización de árboles":
    not_wanted = []
    st.title("Visualizador de centralidad en árboles de dependencias sintácticas")

    GRAPH_DIR = "./UD_Spanish-GSD/"  # Set your desired directory path here
    graph_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]
    #vis_mode = st.sidebar.radio("Visualization mode", ["Centrality", "Syntax Tree"])

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
        centrality = st.sidebar.selectbox("Medida de centralidad", ["Betweenness", "Closeness", "Harmonic", "All-Subgraphs", "PageRank"])

        # Hang tree from most central node
        centrality_scores = compute_centrality(G_undirected, centrality)

        # --- Node color mapping for phrase visualization ---
        cmap = cm.viridis
        values = np.array(list(centrality_scores.values()), dtype=float)
        vmin, vmax = values.min(), values.max()
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        bins = np.quantile(values, np.linspace(0, 1, 11))
        norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        node_colors_dict = {G_undirected.nodes[node].get("form", "UNKNOWN"): cmap(norm(centrality_scores[node])) for node in G_undirected.nodes()}

        # --- Display colored phrase ---
        phrase = G_directed.graph.get("phrase")
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
        colored_phrase = "**Frase**: " + ' '.join(final_phrase)
        st.sidebar.markdown(colored_phrase, unsafe_allow_html=True)

        # Display trees side-by-side instead of vertically
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Syntactic Dependency Tree")
            root_node = str(G_directed.graph.get('root', None))
            pos1 = hierarchy_pos(G_directed, root=root_node, vert_gap=3)
            fig1, ax1 = draw_graph(G_directed, {n: 1 for n in G_directed.nodes()}, "Directed Syntactic Dependency Tree", pos1, 'directed')
            st.pyplot(fig1)

        with col2:
            st.subheader(f"Centrality: {centrality}")
            c = compute_centrality(G_undirected, centrality)
            root_node = max(c, key=c.get)
            G_dag = build_dag_from_root(G_undirected, root_node)
            pos2 = hierarchy_pos(G_dag, root=root_node, vert_gap=3)
            fig2, ax2 = draw_graph(G_undirected, c, f"Centrality: {centrality}", pos2)
            st.pyplot(fig2)

elif view == "Datos de distribución":
    st.title("Visualización de distrubición")
    
    #st.sidebar.markdown("### Subir archivo CSV local")
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
