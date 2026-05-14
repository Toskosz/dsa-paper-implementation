"""Interference conflict graph construction.

Builds the user-centric conflict graph G(t) = (V, E) used by the graph
coloring algorithm.  Each vertex represents a UE, and an edge (u, v)
indicates that UEs u and v cannot share the same PRB due to interference.

Two conditions create a conflict edge:
  1. **Same-ORU conflict:** Both UEs are served by the same O-RU — they
     compete for the same radio resources and must use different PRBs.
  2. **Inter-cell SINR degradation:** UEs on different O-RUs would each
     suffer unacceptable rate loss if assigned the same PRB, because
     one UE's signal becomes the other's interference.

Time complexity of graph construction is O(U²) due to the pairwise scan.
"""

from __future__ import annotations


from src.models import NetworkConfig, UE
from src.propagation import achievable_rate_mbps, sinr_with_sharing


def build_conflict_graph(
    ues: list[UE],
    config: NetworkConfig,
    prb_bw_khz: float,
) -> tuple[dict[int, list[int]], int]:
    """Construct the interference conflict graph.

    Args:
        ues: List of active UEs (graph vertices).
        config: Network configuration with O-RU positions and parameters.
        prb_bw_khz: PRB bandwidth in kHz (for SINR rate computation).

    Returns:
        A tuple (adjacency, edge_count):
        - adjacency: Dict mapping each UE id to a list of conflicting UE ids.
        - edge_count: Total number of undirected conflict edges.
    """
    oru_by_id = {oru.id: oru for oru in config.orus}
    adjacency: dict[int, list[int]] = {ue.id: [] for ue in ues}

    # Pairwise comparison: O(U²) — each pair is checked once
    edge_count = 0
    for i, u1 in enumerate(ues):
        for u2 in ues[i + 1 :]:
            if _has_conflict(u1, u2, config, oru_by_id, prb_bw_khz):
                adjacency[u1.id].append(u2.id)
                adjacency[u2.id].append(u1.id)
                edge_count += 1

    return adjacency, edge_count


def _has_conflict(
    u1: UE,
    u2: UE,
    config: NetworkConfig,
    oru_by_id: dict,
    prb_bw_khz: float,
) -> bool:
    """Determine whether two UEs conflict (cannot share a PRB).

    Conflict conditions:
      1. Same serving O-RU → must use different PRBs.
      2. Different O-RUs but sharing a PRB would drop either UE's
         achievable rate below its satisfaction tolerance margin.
    """
    # Condition 1: co-served users always conflict
    if u1.connected_oru == u2.connected_oru and u1.connected_oru is not None:
        return True

    if u1.connected_oru is None or u2.connected_oru is None:
        return False

    # Condition 2: check if PRB sharing would violate rate requirements
    sinr_u1 = sinr_with_sharing(u1, oru_by_id[u1.connected_oru], config.orus, config.noise_per_prb_dbm)
    sinr_u2 = sinr_with_sharing(u2, oru_by_id[u2.connected_oru], config.orus, config.noise_per_prb_dbm)

    rate_u1 = achievable_rate_mbps(sinr_u1, prb_bw_khz)
    rate_u2 = achievable_rate_mbps(sinr_u2, prb_bw_khz)

    # Minimum acceptable rate = demand × (1 − tolerance)
    min_rate_u1 = u1.demand_mbps * (1.0 - u1.satisfaction_tolerance)
    min_rate_u2 = u2.demand_mbps * (1.0 - u2.satisfaction_tolerance)

    return rate_u1 < min_rate_u1 or rate_u2 < min_rate_u2


def weighted_degrees(ues: list[UE], adjacency: dict[int, list[int]]) -> dict[int, float]:
    """Compute weighted degree for each UE: deg(u) × w_u.

    The weighted degree determines the coloring order — UEs with more
    conflicts and higher priority are colored first (Welsh-Powell heuristic).

    Args:
        ues: List of UEs.
        adjacency: Conflict graph adjacency list.

    Returns:
        Dict mapping UE id to its weighted degree.
    """
    degrees: dict[int, float] = {}
    for ue in ues:
        degrees[ue.id] = len(adjacency.get(ue.id, [])) * ue.priority_weight
    return degrees
