# unordered_tree_edit_distance_nx.py
# ------------------------------------------------------------
# Unordered Tree Edit Distance (TED) for NetworkX trees.
# Supports:
#   - Unlabeled, unordered trees (structure-only)
#   - Labeled, unordered trees (relabel cost)
#
# Edit operations:
#   - insert node   (default cost 1)
#   - delete node   (default cost 1)
#   - relabel node  (default cost 0 if same label, else 1)
#
# Implementation notes:
#   - Exact DP on rooted trees with a Hungarian step (child-set assignment)
#   - For undirected trees you can opt into "unrooted" distance
#     (min over centers) to reduce root-choice bias.
# ------------------------------------------------------------
from __future__ import annotations
from typing import Callable, Iterable, Optional, Any, Union, List, Tuple
from functools import lru_cache

INF = 10**12  # big finite number for "forbidden" assignments


# ---------- Minimal unordered tree structure ----------
class UTree:
    """
    Simple unordered rooted tree. Child order is irrelevant.
    'label' is a string (or None if unlabeled).
    """
    __slots__ = ("label", "children")
    def __init__(self, label: Optional[Any] = None, children: Optional[Iterable['UTree']] = None):
        self.label = None if label is None else str(label)
        self.children: List['UTree'] = list(children or [])


# ---------- Hungarian algorithm (square, minimization) ----------
def _hungarian(cost: List[List[float]]) -> Tuple[float, List[int]]:
    """
    Minimal-cost assignment for a square cost matrix.
    Returns (total_cost, assignment) where assignment[i] = chosen column for row i.
    """
    n = len(cost)
    # potentials
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    # p[j] = row matched to column j
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    total = 0.0
    for j in range(1, n + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    for i in range(n):
        total += cost[i][assignment[i]]
    return total, assignment


# ---------- Core Unordered TED ----------
def _collect_subtree_costs(
    root: UTree,
    cost_del: Callable[[Optional[str]], float],
    cost_ins: Callable[[Optional[str]], float],
):
    """
    Precompute cost to delete and cost to insert the entire subtree rooted at each node.
    del[node] = cost_del(label(node)) + sum(del[child])
    ins[node] = cost_ins(label(node)) + sum(ins[child])
    """
    del_cost = {}
    ins_cost = {}

    def dfs(node: UTree) -> Tuple[float, float]:
        d = cost_del(node.label)
        ins = cost_ins(node.label)
        for ch in node.children:
            cd, ci = dfs(ch)
            d += cd
            ins += ci
        del_cost[node] = d
        ins_cost[node] = ins
        return d, ins

    dfs(root)
    return del_cost, ins_cost


def unordered_tree_edit_distance(
    t1: UTree,
    t2: UTree,
    *,
    use_labels: bool = True,
    cost_del: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_ins: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_sub: Callable[[Optional[str], Optional[str]], float] = lambda a, b: 0.0 if a == b else 1.0,
) -> float:
    """
    Exact unordered tree edit distance between two ROOTED trees.

    If use_labels=False -> structure-only distance (labels ignored, relabel cost = 0).
    Otherwise relabel cost = cost_sub.

    Complexity (rough feel): O(|t1|*|t2|*Δ^3) where Δ is max branching factor.
    """
    del1, ins1 = _collect_subtree_costs(t1, cost_del, cost_ins)
    del2, ins2 = _collect_subtree_costs(t2, cost_del, cost_ins)

    @lru_cache(maxsize=None)
    def d(u: UTree, v: UTree) -> float:
        sub = cost_sub(u.label, v.label) if use_labels else 0.0
        cu, cv = u.children, v.children
        ku, kv = len(cu), len(cv)
        if ku == 0 and kv == 0:
            return sub

        n = ku + kv
        # Build assignment matrix with four blocks:
        #   [ pairwise d(child_u, child_v)   |  deletions (diag)  ]
        #   [ insertions (diag)              |  zeros             ]
        C = [[INF] * n for _ in range(n)]

        # top-left (child-to-child)
        for i, a in enumerate(cu):
            for j, b in enumerate(cv):
                C[i][j] = d(a, b)

        # top-right: delete child a_i (only on its own column)
        for i, a in enumerate(cu):
            C[i][kv + i] = del1[a]

        # bottom-left: insert child b_j (only on its own row)
        for j, b in enumerate(cv):
            C[ku + j][j] = ins2[b]

        # bottom-right: zeros allow unused insert-rows to pair with unused delete-columns
        for i in range(ku, n):
            for j in range(kv, n):
                C[i][j] = 0.0

        match_cost, _ = _hungarian(C)
        return sub + match_cost

    return d(t1, t2)


# ---------- NetworkX adapters & pairwise helpers ----------
def nx_to_unordered_tree(
    G,
    root: Any = None,
    *,
    label_attr: Optional[str] = "label",
) -> UTree:
    """
    Convert a NetworkX tree (Graph or DiGraph) to UTree (unordered).
    - For DiGraph, assumes edges are parent->child if a in-degree-0 root exists;
      if not, tries child->parent; otherwise picks a reasonable node.
    - For Graph, we choose a tree center as root (reduces root bias on undirected).
    - If label_attr is None, labels are ignored (structure-only).
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError("This function requires networkx. `pip install networkx`") from e

    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError("G must be a networkx Graph or DiGraph.")

    directed = isinstance(G, nx.DiGraph)

    if directed:
        use_successors = True
        if root is None:
            roots_in0 = [n for n in G.nodes if G.in_degree(n) == 0]
            if roots_in0:
                root = roots_in0[0]
                use_successors = True
            else:
                roots_out0 = [n for n in G.nodes if G.out_degree(n) == 0]
                if roots_out0:
                    root = roots_out0[0]
                    use_successors = False
                else:
                    root = max(G.nodes, key=lambda n: G.out_degree(n))
                    use_successors = True
        else:
            if G.in_degree(root) == 0:
                use_successors = True
            elif G.out_degree(root) == 0:
                use_successors = False
            else:
                use_successors = True

        nxts = G.successors if use_successors else G.predecessors
        visited = set()

        def build(u, parent=None) -> UTree:
            visited.add(u)
            children = []
            for v in nxts(u):
                if v == parent or v in visited:
                    continue
                children.append(build(v, u))
            label = G.nodes[u].get(label_attr, u) if label_attr is not None else None
            return UTree(label, children)

        return build(root)

    else:
        # undirected: root at a center to reduce bias
        if root is None:
            try:
                centers = __import__("networkx").center(G)
                root = centers[0] if centers else next(iter(G.nodes))
            except Exception:
                root = next(iter(G.nodes))

        visited = {root}
        parent = {root: None}

        def build(u) -> UTree:
            children = []
            for v in G.neighbors(u):
                if v == parent[u] or v in visited:
                    continue
                parent[v] = u
                visited.add(v)
                children.append(build(v))
            label = G.nodes[u].get(label_attr, u) if label_attr is not None else None
            return UTree(label, children)

        return build(root)


def nx_unordered_tree_edit_distance(
    G1,
    G2,
    *,
    label_attr: Optional[str] = "label",
    use_labels: bool = True,
    cost_del: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_ins: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_sub: Callable[[Optional[str], Optional[str]], float] = lambda a, b: 0.0 if a == b else 1.0,
    root1: Any = None,
    root2: Any = None,
) -> float:
    """
    Unordered TED between two NetworkX trees (rooted).
    Set label_attr=None or use_labels=False for unlabeled (structure-only) distance.
    """
    t1 = nx_to_unordered_tree(G1, root=root1, label_attr=label_attr)
    t2 = nx_to_unordered_tree(G2, root=root2, label_attr=label_attr)
    return unordered_tree_edit_distance(
        t1,
        t2,
        use_labels=use_labels,
        cost_del=cost_del,
        cost_ins=cost_ins,
        cost_sub=cost_sub,
    )


def nx_unordered_tree_edit_distance_unrooted(
    G1,
    G2,
    *,
    label_attr: Optional[str] = "label",
    use_labels: bool = True,
    cost_del: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_ins: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_sub: Callable[[Optional[str], Optional[str]], float] = lambda a, b: 0.0 if a == b else 1.0,
) -> float:
    """
    Unrooted variant for undirected trees:
    compute the min distance over (center(G1), center(G2)) choices.
    For DiGraph, this behaves like the rooted version (it will pick in-degree-0 roots).
    """
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError("This function requires networkx. `pip install networkx`") from e

    # get candidate roots (centers) for undirected, in-degree-0 for directed
    def candidate_roots(G):
        if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
            try:
                cs = nx.center(G)
                return cs if cs else list(G.nodes)[:1]
            except Exception:
                return list(G.nodes)[:1]
        else:
            roots = [n for n in G.nodes if G.in_degree(n) == 0]
            return roots if roots else list(G.nodes)[:1]

    roots1 = candidate_roots(G1)
    roots2 = candidate_roots(G2)

    best = None
    for r1 in roots1:
        t1 = nx_to_unordered_tree(G1, root=r1, label_attr=label_attr)
        for r2 in roots2:
            t2 = nx_to_unordered_tree(G2, root=r2, label_attr=label_attr)
            d = unordered_tree_edit_distance(
                t1, t2, use_labels=use_labels, cost_del=cost_del, cost_ins=cost_ins, cost_sub=cost_sub
            )
            best = d if best is None else min(best, d)
    return float(best if best is not None else 0.0)


def nx_pairwise_unordered_ted(
    trees: Iterable,
    *,
    names: Optional[Iterable[str]] = None,
    label_attr: Optional[str] = "label",
    use_labels: bool = True,
    cost_del: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_ins: Callable[[Optional[str]], float] = lambda a: 1.0,
    cost_sub: Callable[[Optional[str], Optional[str]], float] = lambda a, b: 0.0 if a == b else 1.0,
    return_dataframe: bool = True,
    unrooted: bool = False,
):
    """
    Pairwise unordered TED for a dataset of NetworkX trees.
    - Set (label_attr=None or use_labels=False) for unlabeled (structure-only).
    - If unrooted=True and graphs are undirected, it minimizes over tree centers.

    Returns a pandas.DataFrame (if available) else a nested dict.
    """
    try:
        trees_list = list(trees)
    except TypeError:
        raise TypeError("`trees` must be an iterable of NetworkX Graph/DiGraph objects.")

    # names
    if names is None:
        try:
            names = [G.graph.get("phrase", f"T{i}") for i, G in enumerate(trees_list)]
        except Exception:
            names = [f"T{i}" for i in range(len(trees_list))]
    else:
        names = list(names)
        if len(names) != len(trees_list):
            raise ValueError("`names` must have the same length as `trees`.")

    n = len(trees_list)
    mat = [[0.0] * n for _ in range(n)]

    # If not unrooted, pre-convert once for speed
    rooted_utrees: Optional[List[UTree]] = None
    if not unrooted:
        rooted_utrees = [nx_to_unordered_tree(G, root=None, label_attr=label_attr) for G in trees_list]

    for i in range(n):
        for j in range(i + 1, n):
            if unrooted:
                d = nx_unordered_tree_edit_distance_unrooted(
                    trees_list[i],
                    trees_list[j],
                    label_attr=label_attr,
                    use_labels=use_labels,
                    cost_del=cost_del,
                    cost_ins=cost_ins,
                    cost_sub=cost_sub,
                )
            else:
                d = unordered_tree_edit_distance(
                    rooted_utrees[i],
                    rooted_utrees[j],
                    use_labels=use_labels,
                    cost_del=cost_del,
                    cost_ins=cost_ins,
                    cost_sub=cost_sub,
                )
            mat[i][j] = d
            mat[j][i] = d

    if return_dataframe:
        try:
            import pandas as pd
            return pd.DataFrame(mat, index=names, columns=names)
        except Exception:
            pass  # fall back to nested dict

    return {names[i]: {names[j]: mat[i][j] for j in range(n)} for i in range(n)}