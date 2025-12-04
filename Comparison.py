"""
doe voor elke combinatie van 2 oplossingen:
    1. list[list[int]] --> similarity matrix
    2. optimal assignment
    3. compute final similarity
resultaat: kxk matrix
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def jaccard(group_1: list[int], group_2: list[int]) -> float:
    """
    Computes the jaccard similarities of two groups
    """
    return len(set(group_1) & set(group_2)) / len(set(group_1) | set(group_2))


def jaccard_matrix(groups_1: list[list[int]], groups_2: list[list[int]]) -> np.ndarray:
    """
    Builds a matrix of jaccard similarities between all groups of two solutions
    """

    n = max(len(groups_1), len(groups_2))

    M = np.zeros((n, n), dtype=float)
    for i in range(len(groups_1)):
        for j in range(len(groups_2)):
            M[i,j] = jaccard(groups_1[i], groups_2[j])

    return M


def compute_similarity(M: np.ndarray) -> dict:
    row_idx, col_idx = linear_sum_assignment(-M)  # linear sum assignment minimizes, so we invert to maximize similarity
    values = M[row_idx, col_idx]

    return {
        "simlarity": values.mean(),
        "row_idx": list(row_idx),
        "col_idx": list(col_idx)
    }


if __name__ == '__main__':
    import json

    with open("sample_groups.json", 'rt') as f:
        solutions = json.load(f)

    print(compute_similarity(jaccard_matrix(solutions[0], solutions[1])))