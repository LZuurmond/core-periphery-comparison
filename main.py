import hcp_loader
import comparison

import numpy as np
import tqdm
import collections
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


# find most central solution (medoid)
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
clustering_parameter = 0.15  # how different two solutions need to be to be considered a different core

# Convert similarity to distance and only take the upper triangle, as that is all that linkage needs
distances = 1 - sim_matrix[np.triu_indices(n_samples, k=1)]
Z = linkage(distances, method='average')

# Cluster solutions based on similarity
cluster_labels = fcluster(Z, t=clustering_parameter, criterion='distance')
unique_clusters = np.unique(cluster_labels)
cluster_map = {cid: np.where(cluster_labels == cid)[0] for cid in unique_clusters}

if len(unique_clusters) > 1:
    print("Found {len(unique_clusters)} distinct structural modes (clusters of solutions):")
    for cid in unique_clusters:
        members = np.where(cluster_labels == cid)[0]
        cluster_sims = sim_matrix[np.ix_(members, members)].mean()
        print(f"  Mode {cid} (Intra-Sim: {cluster_sims:.4f}): {len(members)} samples (IDs: {members})")
else:
    print(f"Structure appears unimodal at t={clustering_parameter} (one agreed upon core).")


print("\n ---Node level statistics---")
print(cluster_map)

for mode_id, member_indices in cluster_map.items():
    if len(member_indices) < 2:
        print("hi")
        continue

    # isolate sim matrix for the cluster
    cluster_sim_matrix = sim_matrix[np.ix_(member_indices, member_indices)]
    n_members = len(member_indices)

    # find cluster medoid
    cluster_mean_similarities = cluster_sim_matrix.mean(axis=1)
    medoid_idx = np.argmax(cluster_mean_similarities)  # local
    medoid_id = member_indices[medoid_idx]  # global
    cluster_medoid = solutions[medoid_id]

    print(f"\n [MODE {mode_id} ] (Size: {n_samples})")
    print(f"  Cluster medoid ID: {medoid_id} (cluster avg sim: {cluster_mean_similarities[medoid_idx]:.4f})")

    # calculate node-level membership
    num_groups = len(cluster_medoid)
    node_counts = [collections.defaultdict(int) for _ in range(num_groups)]

    for sample_id in member_indices:
        # find mapping FROM sample TO medoid
        sol = solutions[sample_id]
        mapping = comparison.compute_similarity(comparison.jaccard_matrix(sol, cluster_medoid))['mapping']

        for local_grp_idx, nodes in enumerate(sol):
            # Target is the index of the group in the medoid's structure
            target_idx = mapping.get(local_grp_idx)

            # Check if the group maps to a core group in the medoid
            if target_idx is not None and target_idx < num_groups:
                for node in nodes:
                    node_counts[target_idx][node] += 1

    print(f"  Group Coreness (Aligned to Medoid {medoid_id}'s Structure):")
    for g_idx in range(num_groups):
        # Print the actual nodes in this core for reference
        medoid_group = cluster_medoid[g_idx]
        print(f"    Group {g_idx} (Medoid's Base: Nodes {medoid_group}):")

        counts = node_counts[g_idx]
        sorted_nodes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        processed_nodes = []
        for node, count in sorted_nodes:
            prob = count / n_members
            processed_nodes.append(f"{node}: {prob:.0%}")

        print("      " + ", ".join(processed_nodes))

    print("-" * 40)