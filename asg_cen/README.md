# All Subgraphs Centrality implementation

This folder contains the updated implementation for computing All Subgraphs Centrality for a given graph.

* `all_subgraphs_centrality.py`: contains the main algorithm, currently having two stages. The first stage involves computing subgraphs covering semi-partitions in each bag of the tree decomposition of the graph. The second stage involves traversing the tree combining and updating the previous calculations.
* `contraction_counting.py`: contains the computations for an adjacency matrix. This method is called only for graphs of more than 2 nodes, since the other cases can be solved by simple analysis.
* `decomposition.py`: provides a method for building a rich tree decomposition of a graph, i.e. a tree decomposition that also includes the edges of the graph, each belonging to a unique bag of the decomposition. This method also uses the greedy idea of adding each edge to a valid bag with higher number of components. The idea is to maximize the number of disconected components in each bag, in order to simplify the following computations.
* `partition_tools.py`: includes a variety of methods to modify semi-partitions and building their representation.
* `utils.py`: includes methods required to operate on matrices for computing the centrality. This file is used as a dependency of `contraction_counting.py`.