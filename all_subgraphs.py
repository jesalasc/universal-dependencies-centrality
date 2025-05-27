import math
import networkx as nx
from functools import lru_cache


def get_nodes_from_edges(edges):
    """
    Extract unique nodes from a list of edges.
    Each edge is a tuple (u, v, weight).
    """
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    return list(nodes)


def centr_trees(T: nx.Graph):
    """
    Compute the same “recursive centrality” for every node in a tree T,
    returning a dict {node: log2(centrality)}.
    """
    ALL_NODES = frozenset([i for i in T])

    
    @lru_cache(maxsize=None)
    def _rec_centr(edges_key: frozenset, nodes_key: frozenset, root):
        G = nx.Graph()
        G.add_nodes_from(nodes_key)

        for u, v, w in edges_key:
            G.add_edge(u, v, weight=w)

        # Base case: two‐node tree
        if len(nodes_key) == 2:
            u, v = tuple(nodes_key)
            return G[u][v].get('weight', 1) + 1

        # Otherwise, find any neighbor i of root and apply the R-algorithm
        for nbr in G.neighbors(root):
            # Weight on the root–nbr edge
            w = G[root][nbr].get("weight", 1)

            # 1) Remove the edge root–nbr for the “first” recursion
            G1 = G.copy()
            G1.remove_edge(root, nbr)
            edges1 = frozenset((u, v, G1[u][v].get('weight',1)) for u, v in G1.edges())
            nodes1 = nodes_key
            # For the “contract” branch:
            
            
            # 2) Contract nbr into root for the “second” recursion
            #    (we allow a self-loop to collect the merged weight)
            H = nx.contracted_nodes(G, root, nbr, self_loops=True)
            # self-loop weight ends up at (root, root)
            nodes2 = nodes_key - {nbr}
            self_w = H[root][root].get("weight", 0)
            v_factor = self_w / 2

            # clean up H: remove self-loop, then delete node nbr
            H.remove_edge(root, root)
            #H.remove_node(nbr)
            edges2 = frozenset((u, v, H[u][v].get("weight", 1)) for u, v in H.edges())

            # combine recursively
            return (_rec_centr(edges1, nodes1, root)
                        + v_factor * _rec_centr(edges2, nodes2, root))
        # If we found no neighbor, it's an isolated root (shouldn't happen in a tree)
        return 1

    # Initial edge key includes weights if present, else weight=1
    initial_edges = frozenset((u, v, T[u][v].get('weight', 1)) for u, v in T.edges())
    initial_nodes = frozenset(T.nodes())
    return {n: math.log2(_rec_centr(initial_edges, initial_nodes, n))
            for n in T.nodes()}
