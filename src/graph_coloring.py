from __future__ import annotations

import time

from src.models import UE


def policy_guided_coloring(
    sorted_ues: list[UE],
    adjacency: dict[int, list[int]],
    num_prbs: int,
) -> tuple[dict[int, int], list[UE], float]:
    start = time.perf_counter()

    allocation: dict[int, int] = {}
    uncolored: list[UE] = []

    neighbor_colors: dict[int, set[int]] = {ue.id: set() for ue in sorted_ues}

    for ue in sorted_ues:
        prbs_used_by_neighbors = neighbor_colors.get(ue.id, set())
        assigned = False

        for prb in range(num_prbs):
            if prb not in prbs_used_by_neighbors:
                allocation[ue.id] = prb
                ue.assigned_prb = prb
                assigned = True
                for neighbor_id in adjacency.get(ue.id, []):
                    neighbor_colors[neighbor_id].add(prb)
                break

        if not assigned:
            uncolored.append(ue)

    elapsed = (time.perf_counter() - start) * 1000.0
    return allocation, uncolored, elapsed
