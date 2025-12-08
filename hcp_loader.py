import collections
import math


def load_hcp_solutions(filepath) -> list[list[list[int]]]:
    """
    Parses the hcp configs.txt output file, which contains the group assignments
    returns a list of solutions as a list[list[int]
    """

    solutions = []

    with open(filepath, 'r') as f:
        for line in f:
            node_assignments = list(map(int, line.strip().split()))

            max_mask = max(node_assignments) if node_assignments else 0
            max_group_id = int(math.log2(max_mask)) + 1 if max_mask > 0 else 0

            group_dict = collections.defaultdict(list)

            for node_id, group_mask in enumerate(node_assignments):
                for group_id in range(max_group_id):
                    if (group_mask & (1 << group_id)):
                        group_dict[group_id].append(node_id)

            current_solution = [group_dict[k] for k in sorted(group_dict.keys())]
            solutions.append(current_solution)

    return solutions