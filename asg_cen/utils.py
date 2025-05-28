from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} args:[{args[0].__name__}] took: {'%2.4f'%(te-ts)} sec")
        return result

    return wrap


def add_to_row_col(matrix1, matrix2, n, i):
    """
    Adds the ith column and row of matrix2 to matrix1 (in place)
    """
    for k in range(len(matrix1)):
        matrix1[k][n] = matrix1[k][n] + matrix2[k][i]
    for k in range(len(matrix1)):
        matrix1[n][k] = matrix1[n][k] + matrix2[i][k]


def matrix_without_row_col(matrix, n):
    """
    Returns matrix without its nth row and column
    """
    new_matrix = [[0 for _ in range(len(matrix) - 1)] for _ in range(len(matrix) - 1)]
    off_i = 0
    for i in range(len(matrix) - 1):
        off_j = 0
        if i == n:
            off_i = 1
        for j in range(len(matrix) - 1):
            if j == n:
                off_j = 1
            new_matrix[i][j] = matrix[i + off_i][j + off_j]
    return new_matrix


def rec_asgc(matrix, n):
    """
    Computes the number of connected subgraphs of matrix that contain vertex n
    """
    if len(matrix) == 2:
        return 2 ** matrix[0][1]
    for i in range(len(matrix)):
        if i != n and matrix[n][i] != 0:
            mt = [row[:] for row in matrix]
            add_to_row_col(mt, mt, n, i)
            v = mt[n][n] / 2
            mt[n][n] = 0
            mt = matrix_without_row_col(mt, i)
            matrix[n][i] = 0
            matrix[i][n] = 0
            k = int(i < n)
            return rec_asgc(matrix, n) + (2 ** v - 1) * rec_asgc(mt, n - k)
    return 1


def rec_atc(matrix, n):
    """
    Computes the number of subtrees of matrix that contain vertex n
    """
    if len(matrix) == 2:
        return matrix[0][1] + 1
    for i in range(len(matrix)):
        if i != n and matrix[n][i] != 0:
            mt = [row[:] for row in matrix]
            add_to_row_col(mt, mt, n, i)
            v = mt[n][n] / 2
            mt[n][n] = 0
            mt = matrix_without_row_col(mt, i)
            matrix[n][i] = 0
            matrix[i][n] = 0
            k = int(i < n)
            return rec_atc(matrix, n) + v * rec_atc(mt, n - k)
    return 1
