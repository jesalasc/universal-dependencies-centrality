# import psutil
#from main import format_d
import networkx as nx
from networkx.generators.trees import random_tree
import numpy as np
from re import split
from itertools import combinations, product
from collections import defaultdict
from math import pow, log2
#from prettytable import PrettyTable
from time import time
from scipy.io import mmread
#from progress.bar import Bar
from asg_cen.decomposition import get_greedy_rich_tree_decomp
from asg_cen.partition_tools import add_node_to_partition, format_d, get_partition, partition_repr, get_supremum, get_subset_positions, get_not_null_partitions, remove_node_from_partition
from asg_cen.contraction_counting import contraction_subgraph_count

#from main import all_subgraphs_centrality_updated, EXHAUSTIVE


def get_covered_nodes(formated_p):
    return '.'.join(sorted(split('[-.]', formated_p), key=lambda t: int(t)))


def all_subgraphs_centrality(graph, vee=None):
    # print('comienzo')
    # print('\t', psutil.Process().memory_info().rss / (1024 * 1024))
    # computes all subgraph centrality for a single node if vee != None
    # or provides the complete centralities for the entire graph if vee = None

    # uses a greedy approach to build the tree decomposition (add edges to maximize components)
    # and uses the contraction method to compute centralities in the bags of the decomposition
    _t = time()
    rtd = get_greedy_rich_tree_decomp(graph)
    #print('TD ready in', time()-_t)
    initial_values = dict()
    centralities = dict()
    initial_not_null_partitions = dict()
    initial_degrees = dict()
    # I. COMPUTE CENTRALITY IN BAGS
    # bar = Bar('Bags:', max=len(rtd.nodes()),
    #           suffix='%(index)d/%(max)d [%(elapsed_td)s]')
    for u in rtd.nodes():
        initial_degrees[u] = rtd.degree(u)
        initial_not_null_partitions[u] = dict()
        separated_component_values = list()
        # Separate analysis for each component in this bag
        for comp in rtd.nodes()[u]['graph components']:
            separated_component_values.append(dict())
            universe = sorted(comp.nodes(), key=lambda u: int(u))
            component_values = dict()
            for integer_repr in range(1, int(pow(2, len(universe)))):
                node_set = [universe[i]
                            for i in get_subset_positions(integer_repr)]
                node_set.reverse()
                if len(node_set) == 1:
                    component_values[integer_repr] = 1
                    continue
                if len(node_set) == 2:
                    if node_set[0] in rtd.nodes()[u]['subgraph'][node_set[1]]:
                        component_values[integer_repr] = 1
                    continue
                # CHECK if there is a single component
                matrix = [[0 for _ in node_set] for _ in node_set]
                for i in range(0, len(node_set) - 1):
                    for j in range(i + 1, len(node_set)):
                        if node_set[j] in rtd.nodes()[u]['subgraph'][node_set[i]]:
                            matrix[i][j] = 1
                            matrix[j][i] = 1
                m = nx.convert_matrix.from_numpy_matrix(np.array(matrix))
                if not nx.is_connected(m):
                    continue
                val = contraction_subgraph_count(matrix, set([i for i in range(len(node_set))]), [
                                                 [i for i in range(len(node_set))]])
                component_values[integer_repr] = val
            # Build valid partitions for this component
            # e.g. if A and B are subsets of nodes with A inter B = 0, then [A,B] is a valid partition
            for p in get_not_null_partitions(list(component_values.keys())):
                part = [[universe[k]
                         for k in get_subset_positions(i)] for i in p]
                accum = 1
                for i in p:
                    accum *= component_values[i]
                separated_component_values[-1][partition_repr(part)] = accum
        dist = format_d(rtd.nodes()[u]["distinguished"])
        if len(rtd.nodes()[u]['graph components']) == 1:
            for p, value in separated_component_values[-1].items():
                key = f'{u}:{dist}:{p}'
                cov = get_covered_nodes(p)
                if dist not in initial_not_null_partitions[u]:
                    initial_not_null_partitions[u][dist] = dict()
                if cov not in initial_not_null_partitions[u][dist]:
                    initial_not_null_partitions[u][dist][cov] = set()
                initial_not_null_partitions[u][dist][cov].add(p)
                initial_values[key] = value
        else:
            for k in range(1, len(rtd.nodes()[u]['graph components'])+1):
                for selected_indexes in combinations(
                        [i for i in range(len(rtd.nodes()[u]['graph components']))], k):
                    for prod in product(*[separated_component_values[c] for c in selected_indexes]):
                        p = []
                        value = 1
                        for i, s in enumerate(prod):
                            p.extend(get_partition(s))
                            value *= separated_component_values[selected_indexes[i]][s]
                        p = partition_repr(p)
                        key = f'{u}:{dist}:{p}'
                        cov = get_covered_nodes(p)
                        if dist not in initial_not_null_partitions[u]:
                            initial_not_null_partitions[u][dist] = dict()
                        if cov not in initial_not_null_partitions[u][dist]:
                            initial_not_null_partitions[u][dist][cov] = set()
                        initial_not_null_partitions[u][dist][cov].add(p)
                        initial_values[key] = value
    #     bar.next()
    # bar.finish()
    if vee:
        root = next(filter(lambda node: any(
            (vee == e for e in node.split('-')[0].split('.'))), rtd.nodes()), None)
        to_check_tree_nodes = [root]
    else:
        to_check_tree_nodes = []
        for n in graph.nodes():
            bag = next(filter(lambda b: any((str(n) == p
                       for p in b.split('-')[0].split('.'))), rtd.nodes()), None)
            if bag not in to_check_tree_nodes:
                to_check_tree_nodes.append(bag)
    for root in to_check_tree_nodes:
        # Traverse the tree decomposition
        not_null_partitions = dict()
        for k in initial_not_null_partitions:
            not_null_partitions[k] = dict()
            for d in initial_not_null_partitions[k]:
                not_null_partitions[k][d] = dict()
                for c, s in initial_not_null_partitions[k][d].items():
                    not_null_partitions[k][d][c] = set(s)
        values = dict(initial_values)
        degrees = dict(initial_degrees)
        leaves = [u for u in rtd.nodes() if u !=
                  root and degrees[u] == 1]
        x = 1
        while len(leaves) > 0:
            w = leaves.pop()
            u = [n for n in rtd.neighbors(w) if degrees[n] > 0][0]
            dist_w = rtd.nodes()[w]["distinguished"]
            dist_u = rtd.nodes()[u]["distinguished"]
            # I. FORGET from w every node in the following set
            nodes_to_forget = dist_w - dist_u
            original_ref_set = set(dist_w)
            for n in nodes_to_forget:
                new_ref_set = original_ref_set - set([n])
                new_values = defaultdict(int)
                original_d = format_d(original_ref_set)
                new_d = format_d(new_ref_set)
                if original_d in not_null_partitions[w]:
                    for c in not_null_partitions[w][original_d]:
                        for p in not_null_partitions[w][original_d][c]:
                            p_wo_n = remove_node_from_partition(p, n)
                            if not p_wo_n:
                                continue
                            cov = get_covered_nodes(p_wo_n)
                            new_values[f'{w}:{new_d}:{p_wo_n}'] += values[f'{w}:{original_d}:{p}']
                            if new_d not in not_null_partitions[w]:
                                not_null_partitions[w][new_d] = dict()
                            if cov not in not_null_partitions[w][new_d]:
                                not_null_partitions[w][new_d][cov] = set()
                            not_null_partitions[w][new_d][cov].add(p_wo_n)
                original_ref_set.discard(n)
                for k, v in new_values.items():
                    values[k] = v
            # print('======= AFTER FORGET ========')
            # for k, v in values.items():
            #     print(k, v)
            # return

            # II. ADD to w every node in the following set
            nodes_to_add = dist_u - dist_w
            original_ref_set = set(dist_w) - nodes_to_forget
            for n in nodes_to_add:
                new_ref_set = original_ref_set.union(set([n]))
                new_d = format_d(new_ref_set)
                original_d = format_d(original_ref_set)
                new_values = {f'{w}:{new_d}:{partition_repr([[n]])}': 1}
                cov = str(n)
                if new_d not in not_null_partitions[w]:
                    not_null_partitions[w][new_d] = dict()
                    not_null_partitions[w][new_d][cov] = set()
                not_null_partitions[w][new_d][cov].add(partition_repr([[n]]))
                if original_d in not_null_partitions[w]:
                    for c in not_null_partitions[w][original_d]:
                        for p in not_null_partitions[w][original_d][c]:
                            p_w_n = add_node_to_partition(p, n)
                            cov = get_covered_nodes(p_w_n)
                            new_values[f'{w}:{new_d}:{p_w_n}'] = values[f'{w}:{original_d}:{p}']
                            new_values[f'{w}:{new_d}:{p}'] = values[f'{w}:{original_d}:{p}']
                            if cov not in not_null_partitions[w][new_d]:
                                not_null_partitions[w][new_d][cov] = set()
                            if c not in not_null_partitions[w][new_d]:
                                not_null_partitions[w][new_d][c] = set()
                            not_null_partitions[w][new_d][cov].add(p_w_n)
                            not_null_partitions[w][new_d][c].add(p)
                original_ref_set.add(n)
                for k, v in new_values.items():
                    values[k] = v

            # print('======= AFTER ADD ========')
            # for k, v in values.items():
            #     print(k, v)

            # III. Combine leaf and father
            curr_d = format_d(dist_u)
            new_values = defaultdict(int)
            supremums = dict()

            if curr_d in not_null_partitions[u] and curr_d in not_null_partitions[w]:
                for c in not_null_partitions[u][curr_d]:
                    if c not in not_null_partitions[w][curr_d]:
                        continue
                    # bar = Bar(f'{x}/{len(rtd.nodes()) - 1} Combinations:', max=(len(not_null_partitions[u][curr_d][c])*len(not_null_partitions[w][curr_d][c])),
                    #           suffix='%(index)d/%(max)d [%(elapsed_td)s]')
                    for p1 in not_null_partitions[u][curr_d][c]:
                        for p2 in not_null_partitions[w][curr_d][c]:
                            # print(p1, '\t', p2)
                            sup = get_supremum(
                                get_partition(p1), get_partition(p2))
                            if sup:
                                supremum_p = partition_repr(sup)
                                if c not in supremums:
                                    supremums[c] = set()
                                supremums[c].add(supremum_p)
                                new_values[f'{u}:{curr_d}:{supremum_p}'] += values[f'{u}:{curr_d}:{p1}'] * \
                                    values[f'{w}:{curr_d}:{p2}']
                    #             bar.next()
                    # bar.finish()
            if curr_d in not_null_partitions[u]:
                for c in not_null_partitions[u][curr_d]:
                    p_to_forget = []
                    for p in not_null_partitions[u][curr_d][c]:
                        values.pop(f'{u}:{curr_d}:{p}', None)
                        if c not in supremums or p not in supremums[c]:
                            p_to_forget.append(p)
                    for p in p_to_forget:
                        not_null_partitions[u][curr_d][c].remove(p)
            else:
                not_null_partitions[u][curr_d] = dict()
                for c, s in supremums.items():
                    not_null_partitions[u][curr_d][c] = set(s)
            for c, s in supremums.items():
                not_null_partitions[u][curr_d][c].update(s)
            for k, v in new_values.items():
                values[k] = v
            # IV. Remove leaf and update loop
            degrees[w] = 0
            degrees[u] -= 1
            leaves = [u for u in rtd.nodes() if u !=
                      root and degrees[u] == 1]
            x += 1
            # print('=========== AFTER COMBINING ============')
            # for k, v in values.items():
            #     print(k, v)

            # print(w)
            # print('\t', psutil.Process().memory_info().rss / (1024 * 1024))
        # V. Final summary for vee in root
        if vee:
            partial_sum = 0
            ref_set = rtd.nodes()[root]['distinguished']
            curr_d = format_d(ref_set)
            if curr_d in not_null_partitions[root]:
                for c in not_null_partitions[root][curr_d]:
                    for p in not_null_partitions[root][curr_d][c]:
                        pm = [a.split('.') for a in p.split('-')]
                        if len(pm) == 1 and vee in pm[0]:
                            partial_sum += values[f'{root}:{curr_d}:{p}']
            centralities[vee] = partial_sum
        else:
            for v in root.split('-')[0].split('.'):
                if v not in centralities:
                    partial_sum = 0
                    ref_set = rtd.nodes()[root]['distinguished']
                    curr_d = format_d(ref_set)
                    if curr_d in not_null_partitions[root]:
                        for c in not_null_partitions[root][curr_d]:
                            for p in not_null_partitions[root][curr_d][c]:
                                pm = [a.split('.') for a in p.split('-')]
                                if len(pm) == 1 and v in pm[0]:
                                    partial_sum += values[f'{root}:{curr_d}:{p}']
                    centralities[v] = partial_sum
    return centralities[vee] if vee else {k: log2(v) for k, v in centralities.items()}


# if __name__ == '__main__':
#     # g = nx.complete_graph(9)
#     # g = random_tree(200, 1)
#     # g = nx.Graph()
#     # g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
#     g = nx.les_miserables_graph()
#     # g = nx.karate_club_graph()
#     # g = nx.florentine_families_graph()
#     # a = mmread('./soc-wiki-Vote.mtx')
#     # print(a)

#     nodelist = {n: i for i, n in enumerate(g.nodes())}
#     g = nx.relabel_nodes(g, nodelist)
#     value = str(nodelist['Valjean'])

#     x = PrettyTable()
#     x.field_names = ['Method', 'Value', 'Time [s]']
#     x.align['Time [s]'] = 'l'

#     _t = time()
#     td_val = all_subgraphs_centrality(g, vee=value)
#     x.add_row(['Current', td_val, time() - _t])

#     # _t = time()
#     # td_val = all_subgraphs_centrality_updated(g, vee=value, method=EXHAUSTIVE)
#     # x.add_row(['Exhaustive Updated Comp', td_val, time() - _t])

#     print(x)
