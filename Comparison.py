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
import tqdm


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


# %% Analysis
def analyze_ensemble(solutions):
    """
    Analyzes a set of solutions
    :param solutions: list[list[list[int]]] representing the different groups
    """
    n_samples = len(solutions)
    print(f"Analyzing  {n_samples} solutions...")

    # step 1: compute similarity matrix
    sim_matrix = np.zeros((n_samples, n_samples))

    for idx in tqdm.tqdm(range(n_samples * n_samples), desc="Computing pairwise similarities"):
        i = idx // n_samples
        j = idx % n_samples
        res = compute_similarity(jaccard_matrix(solutions[i], solutions[j]))
        sim_matrix[i, j] = res["similarity"]
        sim_matrix[j, i] = res["similarity"]

    # find most central solution
    mean_similarities = sim_matrix.mean(axis=1)
    central_idx = np.argmax(mean_similarities)
    central_solution = solutions[central_idx]

    print("\n--- Structural Stability ---")
    print(f"Overall average similarity: {sim_matrix.mean():.4f}")
    print(f"Most central solution (highest avg similarity with others): {central_idx+1} (Avg Sim: {mean_similarities[central_idx]:.4f})")



if __name__ == '__main__':
    import json

    with open("sample_groups.json", 'rt') as f:
        solutions = json.load(f)

    analyze_ensemble(solutions)