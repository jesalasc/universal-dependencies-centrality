from asg_cen.utils import add_to_row_col, matrix_without_row_col


def has_ended(matrix, partition):
    # we are interested in knowing if there are edges from the partition nodes
    # such that they don't go to other bag
    for a in partition:
        excluded_nodes = list(filter(lambda n: n not in a, (
            n for s in partition for n in s)))
        for n in a:
            for i in range(len(matrix)):
                if matrix[n][i] != 0 and i not in excluded_nodes:
                    return False
    return True


def rec_partition_allsubgraphs(matrix, partition):
    # Receives the matrix without the ignored nodes
    is_ended = has_ended(matrix, partition)
    if all((len(a) == 1 for a in partition)) and is_ended:
        return 1
    if is_ended:
        return 0
    for a in partition:
        n = a[0]
        excluded_nodes = list(filter(lambda v: v not in a, (
            n for s in partition for n in s)))
        for i in range(len(matrix)):
            if matrix[n][i] != 0 and n != i and i not in excluded_nodes:
                mt = [row[:] for row in matrix]
                add_to_row_col(mt, mt, n, i)
                v = mt[n][n] / 2
                mt[n][n] = 0
                mt = matrix_without_row_col(mt, i)
                matrix[n][i] = 0
                matrix[i][n] = 0
                new_partition = [p[:] for p in partition]
                if i in a:
                    idx = new_partition.index(a)
                    new_partition[idx] = list(
                        filter(lambda node: node != i, a))
                for p in new_partition:
                    for j, u in enumerate(p):
                        if i < u:
                            p[j] = u - 1
                f1 = rec_partition_allsubgraphs(matrix, partition)
                f2 = rec_partition_allsubgraphs(mt, new_partition)
                return f1 + (2**v - 1)*f2
    return 0


def contraction_subgraph_count(matrix, distinguished, partition):
    covered = [
        n for s in partition for n in s]
    excluded = list(filter(lambda n: n not in covered, distinguished))
    for n in excluded:
        for i in range(len(matrix)):
            matrix[n][i] = 0
            matrix[i][n] = 0
    while len(excluded) > 0:
        e = excluded.pop()
        matrix = matrix_without_row_col(matrix, e)
        for a in partition:
            for k, n in enumerate(a):
                if e < n:
                    a[k] = n - 1
        for k, _e in enumerate(excluded):
            if e < _e:
                excluded[k] = _e - 1
    return int(rec_partition_allsubgraphs(matrix, partition))
