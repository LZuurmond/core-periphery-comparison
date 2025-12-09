import hcp_loader
import comparison

import numpy as np
import tqdm
from scipy.cluster.hierarchy import linkage, fcluster


# Open file to analyze
with open("sample_groups.json", 'rt') as f:
    solutions = hcp_loader.load_hcp_solutions("hcp-main/hcp_sims/clique_cp_configs.txt")

# get number of solutions
n_samples = len(solutions)
print(f"Analyzing  {n_samples} solutions...")


# step 1: compute similarity matrix
sim_matrix = np.zeros((n_samples, n_samples))

for idx in tqdm.tqdm(range(n_samples * n_samples), desc="Computing pairwise similarities"):
    i = idx // n_samples
    j = idx % n_samples
    res = comparison.compute_similarity(comparison.jaccard_matrix(solutions[i], solutions[j]))
    sim_matrix[i, j] = res["similarity"]
    sim_matrix[j, i] = res["similarity"]


# find most central solution
mean_similarities = sim_matrix.mean(axis=1)
central_idx = np.argmax(mean_similarities)
central_solution = solutions[central_idx]

print("\n--- Structural Stability ---")
print(f"Overall average similarity: {sim_matrix.mean():.4f}")
print(f"Most central solution (highest avg similarity with others): {central_idx + 1} (Avg Sim: {mean_similarities[central_idx]:.4f})")


print("\n ---Multimodality---")
"""
We try to find if there are vastly different solutions for a core and not just node-level differences 
"""
# Convert similarity to distance and only take the upper triangle, as that is all that linkage needs
distances = 1 - sim_matrix[np.triu_indices(n_samples, k=1)]
Z = linkage(distances, method='average')

# Cluster solutions based on similarity. t=0.15 is the clustering parameter for how different two solutions need to be
cluster_labels = fcluster(Z, t=0.15, criterion='distance')
unique_clusters = np.unique(cluster_labels)

if len(unique_clusters) > 1:
    print("Found {len(unique_clusters)} distinct structural modes (clusters of solutions):")
    for cid in unique_clusters:
        members = np.where(cluster_labels == cid)[0]
        cluster_sims = sim_matrix[np.ix_(members, members)].mean()
        print(f"  Mode {cid} (Intra-Sim: {cluster_sims:.4f}): {len(members)} samples (IDs: {members})")
else:
    print("Structure appears unimodal (one agreed upon core).")


print("\n ---Node level statistics---")
