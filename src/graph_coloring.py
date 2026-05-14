"""Algorithm 1: Policy-Guided Graph Coloring.

Implements the Welsh-Powell-style greedy graph coloring heuristic used
by the xApp to assign Physical Resource Blocks (PRBs) to User Equipment.
PRBs act as "colors" and conflict edges prevent interfering UEs from
receiving the same PRB.

Algorithm outline:
  1. UEs are pre-sorted by weighted degree (highest conflict first).
  2. For each UE, greedily pick the lowest-indexed PRB not used by
     any of its already-colored neighbors.
  3. UEs that cannot be colored (all PRBs blocked by neighbors) are
     returned as "uncolored" for the MPF scheduling fallback.

Complexity:
  - Best case:  Ω(U log U) — sorting dominates when the graph is sparse.
  - Worst case: O(U²) — dense/complete graph forces many neighbor checks.
  - Average case: O(U log U + E) — sorting + traversing actual edges.
"""

from __future__ import annotations

import time

from src.models import UE


def policy_guided_coloring(
    sorted_ues: list[UE],
    adjacency: dict[int, list[int]],
    num_prbs: int,
) -> tuple[dict[int, int], list[UE], float]:
    """Assign PRBs to UEs via greedy graph coloring.

    Iterates through UEs in descending weighted-degree order.  For each
    UE, selects the first available PRB (color) that does not conflict
    with any already-colored neighbor.  The neighbor-color bookkeeping
    is maintained incrementally to avoid repeated graph traversals.

    Args:
        sorted_ues: UEs pre-sorted by weighted degree (descending).
        adjacency: Conflict graph adjacency list (UE id → neighbor ids).
        num_prbs: Number of available PRBs (colors).

    Returns:
        A tuple (allocation, uncolored, elapsed_ms):
        - allocation: Dict mapping UE id → assigned PRB index.
        - uncolored: List of UEs that could not be colored.
        - elapsed_ms: Wall-clock execution time in milliseconds.
    """
    start = time.perf_counter()

    allocation: dict[int, int] = {}
    uncolored: list[UE] = []

    # Track which colors each UE's neighbors have claimed, so we can
    # check color availability in O(degree) instead of scanning all UEs.
    neighbor_colors: dict[int, set[int]] = {ue.id: set() for ue in sorted_ues}

    for ue in sorted_ues:
        prbs_used_by_neighbors = neighbor_colors.get(ue.id, set())
        assigned = False

        # Greedy: pick the first PRB not used by any neighbor
        for prb in range(num_prbs):
            if prb not in prbs_used_by_neighbors:
                allocation[ue.id] = prb
                ue.assigned_prb = prb
                assigned = True
                # Propagate this color to all neighbors' forbidden sets
                for neighbor_id in adjacency.get(ue.id, []):
                    neighbor_colors[neighbor_id].add(prb)
                break

        if not assigned:
            # All PRBs blocked by neighbors — will be handled by MPF scheduler
            uncolored.append(ue)

    elapsed = (time.perf_counter() - start) * 1000.0
    return allocation, uncolored, elapsed
