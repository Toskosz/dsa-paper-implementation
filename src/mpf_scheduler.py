"""Algorithm 2: Modified Proportional Fair (MPF) Scheduling.

Handles UEs that could not be colored by the greedy graph coloring
algorithm.  Uses a proportional-fair metric to decide whether an
uncolored UE should preempt (steal) a PRB from a currently-served UE.

MPF metric for UE u on PRB p:
    M_u(t) = (R_{u,p}(t) / R̄_u(t)) · w_u

where R_{u,p} is the instantaneous rate and R̄_u is the EWMA average.
A higher ratio means the UE is under-served relative to its history,
so the scheduler favours starving users — this is the "proport fair"
property that prevents indefinite starvation.
"""

from __future__ import annotations

from src.models import UE, NetworkConfig
from src.propagation import achievable_rate_mbps, sinr_with_sharing, update_ewma


def mpf_metric(ue: UE, prb: int, config: NetworkConfig, prb_bw_khz: float) -> float:
    """Compute the MPF scheduling metric for a UE on a given PRB.

    M_u(t) = (R_{u,p}(t) / R̄_u(t)) · w_u

    When the EWMA throughput is zero (first scheduling slot for this UE),
    the metric reduces to rate × weight to avoid division by zero.

    Args:
        ue: The user equipment.
        prb: Candidate PRB index (unused in rate calculation but reserved
             for per-PRB channel modeling in future extensions).
        config: Network configuration.
        prb_bw_khz: PRB bandwidth in kHz.

    Returns:
        The MPF metric value (higher = more deserving of this PRB).
    """
    if ue.connected_oru is None:
        return 0.0
    oru_by_id = {oru.id: oru for oru in config.orus}
    serving = oru_by_id[ue.connected_oru]
    sinr = sinr_with_sharing(ue, serving, config.orus, config.noise_per_prb_dbm)
    rate = achievable_rate_mbps(sinr, prb_bw_khz)
    if ue.ewma_throughput <= 0:
        # No history yet — use rate × weight as a bootstrap metric
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
    """Assign PRBs to uncolored UEs via the MPF preemption mechanism.

    For each uncolored UE, the scheduler:
      1. Finds the conflict-free PRB with the highest MPF metric.
      2. If no conflict-free PRB exists, considers all PRBs (allowing
         preemption of currently-served UEs).
      3. If the uncolored UE's metric exceeds the current owner's metric,
         the PRB is stolen (preempted) and reassigned.

    After all assignments, EWMA throughput is updated for every UE.

    Args:
        uncolored: UEs that failed to receive a PRB from graph coloring.
        colored_ues: UEs already assigned a PRB by graph coloring.
        allocation: Current PRB allocation (UE id → PRB index), modified in place.
        adjacency: Conflict graph adjacency list.
        num_prbs: Total number of available PRBs.
        config: Network configuration.
        prb_bw_khz: PRB bandwidth in kHz.

    Returns:
        Updated allocation dict (also modified in place).
    """
    # Build reverse lookup: PRB → owning UE id and its current metric
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

        # Pass 1: try to find a conflict-free PRB (no neighbor uses it)
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

        # Pass 2: if no conflict-free PRB found, allow preemption
        if best_prb == -1:
            for prb in range(num_prbs):
                metric = mpf_metric(ue, prb, config, prb_bw_khz)
                if metric > best_metric:
                    best_metric = metric
                    best_prb = prb

        if best_prb == -1:
            continue

        # If the chosen PRB is already owned, preempt if our metric is higher
        if best_prb in prb_owner:
            owner_id = prb_owner[best_prb]
            owner_metric = prb_metric.get(best_prb, 0.0)
            if best_metric > owner_metric:
                # Preempt: evict the old owner
                old_owner = ue_by_id.get(owner_id)
                if old_owner is not None:
                    old_owner.assigned_prb = None
                    del allocation[owner_id]

                ue.assigned_prb = best_prb
                allocation[ue.id] = best_prb
                prb_owner[best_prb] = ue.id
                prb_metric[best_prb] = best_metric
        else:
            # Free PRB — assign directly
            ue.assigned_prb = best_prb
            allocation[ue.id] = best_prb
            prb_owner[best_prb] = ue.id
            prb_metric[best_prb] = best_metric

    # Update EWMA throughput for all UEs after this scheduling round
    for ue in all_ues:
        if ue.assigned_prb is not None:
            if ue.connected_oru is not None:
                oru_by_id = {oru.id: oru for oru in config.orus}
                serving = oru_by_id[ue.connected_oru]
                sinr = sinr_with_sharing(ue, serving, config.orus, config.noise_per_prb_dbm)
                ue.instant_rate = achievable_rate_mbps(sinr, prb_bw_khz)
        update_ewma(ue, config.ewma_alpha)

    return allocation


def _find_ue(ues: list[UE], ue_id: int) -> UE:
    """Look up a UE by its id in a list. Raises ValueError if not found."""
    for ue in ues:
        if ue.id == ue_id:
            return ue
    raise ValueError(f"UE {ue_id} not found")
