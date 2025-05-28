def get_supremum(p1, p2):
    s1 = [set(a) for a in p1]
    s2 = [set(a) for a in p2]
    # s1_elements = set([i for s in s1 for i in s])
    # s2_elements = set([i for s in s2 for i in s])
    # if len(s1_elements) != len(s2_elements) or len(s1_elements.intersection(s2_elements)) != len(s1_elements):
    #     return None
    new_s = s1
    for b in s2:
        for i, a in enumerate(s1):
            if len(a.intersection(b)):
                new_s[i] = a.union(b)
    if len(new_s) == 1:
        return [list(s) for s in new_s]
    initial_length = len(new_s)
    final_length = len(new_s) + 1
    while initial_length != final_length:
        initial_length = len(new_s)
        i = 0
        break_w = False
        while i < len(new_s) - 1 and not break_w:
            j = i + 1
            while j < len(new_s):
                a = new_s[i]
                b = new_s[j]
                if len(a.intersection(b)):
                    new_s.pop(j)
                    new_s[i] = a.union(b)
                    break_w = True
                    break
                j += 1
            i += 1
        final_length = len(new_s)
    return [list(s) for s in new_s]


def get_not_null_partitions(list_collection):
    yield (list_collection[0],)
    if len(list_collection) == 1:
        return
    for p in get_not_null_partitions(list_collection[1:]):
        if all((list_collection[0] & e == 0 for e in p)):
            yield (list_collection[0], *p)
        yield p


def get_subset_positions(integer_repr):
    bin_repr = bin(integer_repr)
    for i, b in enumerate(bin_repr[2:]):
        if b == '1':
            yield len(bin_repr) - i - 3


def partition_repr(sets):
    return '-'.join(sorted(['.'.join([str(i) for i in sorted(l)]) for l in sorted(sets)]))


def format_d(d_set):
    return ".".join(sorted(list(d_set)))


def get_partition(formated_p):
    return [[int(i) for i in a.split('.')] for a in formated_p.split('-')]


def remove_node_from_partition(formated_p, node):
    p = get_partition(formated_p)
    if p == [[int(node)]]:
        return False
    for a in p:
        if int(node) in a:
            a.remove(int(node))
    if any((len(a) == 0 for a in p)):
        return False
    return partition_repr(p)


def add_node_to_partition(formated_p, node):
    p = get_partition(formated_p)
    p.append([int(node)])
    return partition_repr(p)
