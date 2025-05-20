import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

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
    elif method == "Degree":
        return nx.degree_centrality(G)
    elif method == "Closeness":
        return nx.closeness_centrality(G)
    elif method == "Eigenvector":
        try:
            return nx.eigenvector_centrality(G, max_iter=1000)
        except:
            st.warning("Eigenvector centrality failed to converge.")
            return {n: 0 for n in G.nodes()}
    else:
        return {n: 0 for n in G.nodes()}

# --- Draw Graph ---
def draw_graph(G, centrality, title, pos):
    fig, ax = plt.subplots(figsize=(8, 10))  # Fixed figure size

    values = np.array(list(centrality.values()))
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    node_colors = cmap(norm(values))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Draw labels with black font color and larger font size
    labels = {n: G.nodes[n].get('form', n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black', ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, label='Centrality Score')

    ax.set_title(title)
    ax.axis('off')

    return fig, ax

def hierarchy_pos(G, root=None, width=1.0, vert_gap=2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {}
    if root is None:
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        if not roots:
            raise ValueError("No root found")
        root = roots[0]

    def subtree_width(node):
        children = list(G.neighbors(node))
        if parent is not None:
            children = [c for c in children if c != parent]
        if not children:
            return 1
        return sum(subtree_width(c) for c in children)

    children = list(G.neighbors(root))
    if parent is not None:
        children = [c for c in children if c != parent]

    total_width = sum(subtree_width(c) for c in children)
    dx = width / max(total_width, 1)
    nextx = xcenter - width / 2

    for child in children:
        child_width = subtree_width(child)
        midx = nextx + dx * child_width / 2
        pos = hierarchy_pos(G, root=child, width=dx * child_width, vert_gap=vert_gap,
                            vert_loc=vert_loc - vert_gap, xcenter=midx, pos=pos, parent=root)
        nextx += dx * child_width

    pos[root] = (xcenter, vert_loc)
    return pos

# --- Upload Files ---
st.sidebar.header("Upload GraphML Files")
uploaded_files = st.sidebar.file_uploader("Upload one or more .graphml files", type=["graphml"], accept_multiple_files=True)

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    selected_file_name = st.sidebar.selectbox("Select graph file", file_names)
    selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
    G_directed, G_undirected = load_graph(selected_file)
    pos = hierarchy_pos(G_directed)



    st.sidebar.markdown("### Comparacion de centralidad")
    centrality_1 = st.sidebar.selectbox("Centrality Measure 1", ["Betweenness", "Degree", "Closeness", "Eigenvector"])
    centrality_2 = st.sidebar.selectbox("Centrality Measure 2", ["Betweenness", "Degree", "Closeness", "Eigenvector"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{selected_file_name} - {centrality_1}")
        c1 = compute_centrality(G_undirected, centrality_1)
        fig1, ax1 = draw_graph(G_undirected, c1, f"Centrality: {centrality_1}", pos)
        st.pyplot(fig1)
        st.markdown(f"**Frase**: {G_directed.graph['phrase']}")

    with col2:
        st.subheader(f"{selected_file_name} - {centrality_2}")
        c2 = compute_centrality(G_undirected, centrality_2)
        fig2, ax2 = draw_graph(G_undirected, c2, f"Centrality: {centrality_2}", pos)
        st.pyplot(fig2)
        st.markdown(f"**Frase**: {G_directed.graph['phrase']}")
        
else:
    st.info("Upload one or more `.graphml` files to begin.")