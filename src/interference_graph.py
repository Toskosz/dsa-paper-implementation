from __future__ import annotations


from src.models import NetworkConfig, UE
from src.propagation import achievable_rate_mbps, sinr_with_sharing


def build_conflict_graph(
    ues: list[UE],
    config: NetworkConfig,
    prb_bw_khz: float,
) -> tuple[dict[int, list[int]], int]:
    oru_by_id = {oru.id: oru for oru in config.orus}
    adjacency: dict[int, list[int]] = {ue.id: [] for ue in ues}

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
    if u1.connected_oru == u2.connected_oru and u1.connected_oru is not None:
        return True

    if u1.connected_oru is None or u2.connected_oru is None:
        return False

    sinr_u1 = sinr_with_sharing(u1, oru_by_id[u1.connected_oru], config.orus, config.noise_power_dbm)
    sinr_u2 = sinr_with_sharing(u2, oru_by_id[u2.connected_oru], config.orus, config.noise_power_dbm)

    rate_u1 = achievable_rate_mbps(sinr_u1, prb_bw_khz)
    rate_u2 = achievable_rate_mbps(sinr_u2, prb_bw_khz)

    min_rate_u1 = u1.demand_mbps * (1.0 - u1.satisfaction_tolerance)
    min_rate_u2 = u2.demand_mbps * (1.0 - u2.satisfaction_tolerance)

    return rate_u1 < min_rate_u1 or rate_u2 < min_rate_u2


def weighted_degrees(ues: list[UE], adjacency: dict[int, list[int]]) -> dict[int, float]:
    degrees: dict[int, float] = {}
    for ue in ues:
        degrees[ue.id] = len(adjacency.get(ue.id, [])) * ue.priority_weight
    return degrees
