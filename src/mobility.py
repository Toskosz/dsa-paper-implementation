"""User Equipment mobility model.

Implements a 2D random walk with boundary reflection and O-RU
reassociation.  UEs move at a constant speed in a uniformly random
direction at each time step.  When a UE exits its serving O-RU's
coverage area, it is reflected back inside and reassociated to the
strongest available O-RU.
"""

from __future__ import annotations

import numpy as np

from src.models import ORU, UE
from src.propagation import find_serving_oru


def random_walk_step(ue: UE, speed_mps: float, dt: float = 1.0) -> None:
    """Move a UE one step in a random direction at constant speed.

    A new heading angle θ ∈ [0, 2π) is drawn uniformly each step,
    producing a discrete-time 2D random walk.

    Args:
        ue: The user equipment to move.
        speed_mps: Constant movement speed in m/s.
        dt: Time step duration in seconds.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    ue.vx = speed_mps * np.cos(angle)
    ue.vy = speed_mps * np.sin(angle)
    ue.x += ue.vx * dt
    ue.y += ue.vy * dt


def reflect_in_coverage(ue: UE, orus: list[ORU]) -> None:
    """Reflect a UE back inside its serving O-RU's coverage radius.

    If the UE has moved beyond the coverage boundary, its position is
    projected back to 99% of the radius along the same radial direction.
    This avoids UEs escaping the network topology.
    """
    if ue.connected_oru is None:
        return
    oru = next((o for o in orus if o.id == ue.connected_oru), None)
    if oru is None:
        return
    dist = np.sqrt((ue.x - oru.x) ** 2 + (ue.y - oru.y) ** 2)
    if dist > oru.coverage_radius:
        dx = ue.x - oru.x
        dy = ue.y - oru.y
        # Project back to 99% of the radius along the same direction
        norm = dist
        ue.x = oru.x + dx / norm * oru.coverage_radius * 0.99
        ue.y = oru.y + dy / norm * oru.coverage_radius * 0.99


def reassociate(ue: UE, orus: list[ORU]) -> None:
    """Reassociate the UE to the strongest available O-RU.

    Called after each mobility step so the UE always connects to the
    O-RU offering the best received power at its new position.
    """
    serving = find_serving_oru(ue, orus)
    ue.connected_oru = serving.id


def spawn_ue_in_coverage(
    ue_id: int, orus: list[ORU], demand_mbps: float | None = None
) -> UE:
    """Spawn a UE at a uniformly random position inside a random O-RU's coverage.

    Uses rejection-free polar sampling: r = R·√(U) where U ∼ Uniform(0,1)
    ensures uniform area distribution (not just uniform angle/radius).

    Args:
        ue_id: Unique identifier for the new UE.
        orus: List of O-RUs to choose from.
        demand_mbps: Requested data rate; randomly chosen if None.

    Returns:
        A new UE instance inside the selected O-RU's coverage area.
    """
    oru = orus[np.random.randint(len(orus))]
    angle = np.random.uniform(0, 2 * np.pi)
    # √(uniform) ensures uniform distribution over the circular area
    r = oru.coverage_radius * np.sqrt(np.random.uniform(0, 1))
    x = oru.x + r * np.cos(angle)
    y = oru.y + r * np.sin(angle)
    if demand_mbps is None:
        demand_mbps = np.random.choice([0.5, 1.0, 1.5])
    return UE(
        id=ue_id,
        x=x,
        y=y,
        connected_oru=oru.id,
        demand_mbps=demand_mbps,
    )


def spawn_ue_at_position(ue_id: int, x: float, y: float, orus: list[ORU]) -> UE:
    """Spawn a UE at an exact position and associate it to the best O-RU.

    Used in Phase 2 (worst case) to colocate all UEs at the origin,
    creating a dense conflict graph.

    Args:
        ue_id: Unique identifier for the new UE.
        x, y: Exact spawn coordinates in meters.
        orus: List of O-RUs for association.

    Returns:
        A new UE instance at the specified position.
    """
    serving = find_serving_oru(UE(id=ue_id, x=x, y=y), orus)
    demand_mbps = np.random.choice([0.5, 1.0, 1.5])
    return UE(
        id=ue_id,
        x=x,
        y=y,
        connected_oru=serving.id,
        demand_mbps=demand_mbps,
    )
