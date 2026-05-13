from __future__ import annotations

from src.models import UE, NetworkConfig
from src.propagation import achievable_rate_mbps, sinr_with_sharing, update_ewma


def mpf_metric(ue: UE, prb: int, config: NetworkConfig, prb_bw_khz: float) -> float:
    if ue.connected_oru is None:
        return 0.0
    oru_by_id = {oru.id: oru for oru in config.orus}
    serving = oru_by_id[ue.connected_oru]
    sinr = sinr_with_sharing(ue, serving, config.orus, config.noise_power_dbm)
    rate = achievable_rate_mbps(sinr, prb_bw_khz)
    if ue.ewma_throughput <= 0:
        return rate * ue.priority_weight
    return (rate / ue.ewma_throughput) * ue.priority_weight


def mpf_schedule(
    uncolored: list[UE],
    colored_ues: list[UE],
    allocation: dict[int, int],
    adjacency: dict[int, list[int]],
    num_prbs: int,
    config: NetworkConfig,
    prb_bw_khz: float,
) -> dict[int, int]:
    prb_owner: dict[int, int] = {}
    prb_metric: dict[int, float] = {}

    for ue_id, prb in allocation.items():
        prb_owner[prb] = ue_id
        prb_metric[prb] = mpf_metric(
            _find_ue(colored_ues, ue_id), prb, config, prb_bw_khz
        )

    all_ues = colored_ues + uncolored
    ue_by_id = {ue.id: ue for ue in all_ues}

    for ue in uncolored:
        best_prb = -1
        best_metric = -1.0

        for prb in range(num_prbs):
            neighbor_ids = adjacency.get(ue.id, [])
            conflict = False
            for nid in neighbor_ids:
                if nid in allocation and allocation[nid] == prb:
                    conflict = True
                    break
            if conflict:
                continue

            metric = mpf_metric(ue, prb, config, prb_bw_khz)
            if metric > best_metric:
                best_metric = metric
                best_prb = prb

        if best_prb == -1:
            for prb in range(num_prbs):
                metric = mpf_metric(ue, prb, config, prb_bw_khz)
                if metric > best_metric:
                    best_metric = metric
                    best_prb = prb

        if best_prb == -1:
            continue

        if best_prb in prb_owner:
            owner_id = prb_owner[best_prb]
            owner_metric = prb_metric.get(best_prb, 0.0)
            if best_metric > owner_metric:
                old_owner = ue_by_id.get(owner_id)
                if old_owner is not None:
                    old_owner.assigned_prb = None
                    del allocation[owner_id]

                ue.assigned_prb = best_prb
                allocation[ue.id] = best_prb
                prb_owner[best_prb] = ue.id
                prb_metric[best_prb] = best_metric
        else:
            ue.assigned_prb = best_prb
            allocation[ue.id] = best_prb
            prb_owner[best_prb] = ue.id
            prb_metric[best_prb] = best_metric

    for ue in all_ues:
        if ue.assigned_prb is not None:
            if ue.connected_oru is not None:
                oru_by_id = {oru.id: oru for oru in config.orus}
                serving = oru_by_id[ue.connected_oru]
                sinr = sinr_with_sharing(ue, serving, config.orus, config.noise_power_dbm)
                ue.instant_rate = achievable_rate_mbps(sinr, prb_bw_khz)
        update_ewma(ue, config.ewma_alpha)

    return allocation


def _find_ue(ues: list[UE], ue_id: int) -> UE:
    for ue in ues:
        if ue.id == ue_id:
            return ue
    raise ValueError(f"UE {ue_id} not found")
