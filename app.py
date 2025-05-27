import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from spacy import displacy
import streamlit.components.v1 as components
import os

import sys
# Ensure the all_subgraphs code directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../all_subgraphs_implementation/code")))
from all_subgraphs_centrality import all_subgraphs_centrality as asg

st.set_page_config(layout="wide")
st.title("GraphML Centrality Visualizer")

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
            print({G.nodes[n].get('form', n): asg_cen[n] for n in G.nodes()})  # Debugging output
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
    labels = {n: G.nodes[n].get('form', n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color='red', ax=ax)
    
    if type == 'undirected':
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.9, label='Centrality Score')

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

GRAPH_DIR = "../data/graphs/UD_Spanish-GSD"  # Set your desired directory path here
graph_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".graphml")]
#vis_mode = st.sidebar.radio("Visualization mode", ["Centrality", "Syntax Tree"])

if not graph_files:
    st.error(f"No .graphml files found in {GRAPH_DIR}.")
else:
    selected_file_name = st.sidebar.selectbox("Select graph file", graph_files)
    with open(os.path.join(GRAPH_DIR, selected_file_name), 'r', encoding='utf-8') as f:
        G_directed, G_undirected = load_graph(f)

    pos = hierarchy_pos(G_directed, vert_gap=3)

    st.sidebar.markdown("### Elige una medida de centralidad")
    centrality = st.sidebar.selectbox("Medida de centralidad", ["Betweenness", "Closeness", "Harmonic", "All-Subgraphs", "PageRank"])
    #centrality_2 = st.sidebar.selectbox("Centrality Measure 2", ["Betweenness", "Degree", "Closeness", "Eigenvector"])

    # Arrange figures vertically for better horizontal space
    st.subheader(f"{selected_file_name} - Directed Structure")
    fig1, ax1 = draw_graph(G_directed, {n: 1 for n in G_directed.nodes()}, "Directed Syntactic Dependency Tree", pos, 'directed')
    st.pyplot(fig1)
    st.markdown(f"**Frase**: {G_directed.graph['phrase']}")

    #st.subheader(f"{selected_file_name} - {centrality_2}")
    c = compute_centrality(G_undirected, centrality)
    fig2, ax2 = draw_graph(G_undirected, c, f"Centrality: {centrality}", pos)
    st.pyplot(fig2)
    st.markdown(f"**Frase**: {G_directed.graph['phrase']}")

    # else:  # Syntax Tree mode
    #     words = [{"text": G_directed.nodes[n]["form"]} for n in G_directed.nodes()]
    #     arcs = []
    #     for src, tgt in G_directed.edges():
    #         start = list(G_directed.nodes).index(src)
    #         end = list(G_directed.nodes).index(tgt)
    #         label = G_directed.edges[src, tgt].get("deprel", "")
    #         arcs.append({
    #             "start": min(start, end),
    #             "end": max(start, end),
    #             "label": label,
    #             "dir": "left" if start > end else "right"
    #         })

    #     html = displacy.render({"words": words, "arcs": arcs}, style="dep", manual=True)
    #     components.html(html, height=300, scrolling=True)