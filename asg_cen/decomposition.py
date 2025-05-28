from networkx.algorithms.approximation.treewidth import treewidth_min_degree
import networkx as nx


def get_greedy_rich_tree_decomp(G):
    _, td = treewidth_min_degree(G)
    edges = [('.'.join([str(i) for i in e[0]]), '.'.join([str(i)
              for i in e[1]])) for e in td.edges]
    if len(edges):
        rtd = nx.Graph(edges)
    else:
        rtd = nx.Graph()
        rtd.add_node('.'.join([str(n) for n in G.nodes()]))
    for n in rtd.nodes():
        components = [set([u]) for u in n.split('.')]
        rtd.nodes()[n]['components'] = components
        rtd.nodes()[n]['distinguished'] = set(n.split('.'))
        rtd.nodes()[n]['subgraph'] = nx.Graph()
        rtd.nodes()[n]['subgraph'].add_nodes_from(
            rtd.nodes()[n]['distinguished'])
    node_list = [n for n in rtd.nodes()]
    for e in G.edges():
        node_list = sorted(node_list, key=lambda n: len(
            rtd.nodes()[n]['components']), reverse=True)
        for n in node_list:
            if str(e[0]) in rtd.nodes()[n]['distinguished'] and str(e[1]) in rtd.nodes()[n]['distinguished']:
                comp1 = next(filter(lambda comp: str(
                    e[0]) in comp, rtd.nodes()[n]['components']), None)
                comp2 = next(filter(lambda comp: str(
                    e[1]) in comp, rtd.nodes()[n]['components']), None)
                if comp1 and comp2 and comp1 != comp2:
                    comp1 = rtd.nodes()[n]['components'].index(comp1)
                    comp2 = rtd.nodes()[n]['components'].index(comp2)
                    rtd.nodes()[n]['components'][comp1] = rtd.nodes()[n]['components'][comp1].union(
                        rtd.nodes()[n]['components'][comp2])
                    rtd.nodes()[n]['components'].pop(comp2)
                rtd.nodes()[n]['subgraph'].add_edges_from(
                    [(str(e[0]), str(e[1]))])
                break
    for n in rtd.nodes():
        # each component starts as an empty graph to which we add nodes and edges
        rtd.nodes()[n]['graph components'] = [nx.Graph()
                                              for _ in rtd.nodes()[n]['components']]
        for c, comp in enumerate(rtd.nodes()[n]['components']):
            rtd.nodes()[n]['graph components'][c].add_nodes_from(comp)
        for e in rtd.nodes()[n]['subgraph'].edges():
            for c, comp in enumerate(rtd.nodes()[n]['components']):
                if e[0] in comp:
                    rtd.nodes()[n]['graph components'][c].add_edge(*e)
        if len(rtd.nodes()[n]['subgraph'].edges()):
            edges = sorted([sorted(pair)
                           for pair in rtd.nodes()[n]['subgraph'].edges()])
            edges_string = '-'.join(('.'.join(pair)
                                    for pair in edges))
            rtd = nx.relabel_nodes(rtd, {n: n + '-' + edges_string})
    return rtd
