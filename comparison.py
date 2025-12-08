"""
doe voor elke combinatie van 2 oplossingen:
    1. list[list[int]] --> similarity matrix
    2. optimal assignment
    3. compute final similarity
resultaat: kxk matrix
"""
# %% imports
import numpy as np
from scipy.optimize import linear_sum_assignment


# %% similarity computation
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
    score = M[row_idx, col_idx].mean()

    return {
        "similarity": score,
        "mapping": dict(zip(col_idx, row_idx))
    }

