from __future__ import annotations

import numpy as np

from src.models import ORU, UE
from src.propagation import find_serving_oru


def random_walk_step(ue: UE, speed_mps: float, dt: float = 1.0) -> None:
    angle = np.random.uniform(0, 2 * np.pi)
    ue.vx = speed_mps * np.cos(angle)
    ue.vy = speed_mps * np.sin(angle)
    ue.x += ue.vx * dt
    ue.y += ue.vy * dt


def reflect_in_coverage(ue: UE, orus: list[ORU]) -> None:
    if ue.connected_oru is None:
        return
    oru = next((o for o in orus if o.id == ue.connected_oru), None)
    if oru is None:
        return
    dist = np.sqrt((ue.x - oru.x) ** 2 + (ue.y - oru.y) ** 2)
    if dist > oru.coverage_radius:
        dx = ue.x - oru.x
        dy = ue.y - oru.y
        norm = dist
        ue.x = oru.x + dx / norm * oru.coverage_radius * 0.99
        ue.y = oru.y + dy / norm * oru.coverage_radius * 0.99


def reassociate(ue: UE, orus: list[ORU]) -> None:
    serving = find_serving_oru(ue, orus)
    ue.connected_oru = serving.id


def spawn_ue_in_coverage(
    ue_id: int, orus: list[ORU], demand_mbps: float | None = None
) -> UE:
    oru = orus[np.random.randint(len(orus))]
    angle = np.random.uniform(0, 2 * np.pi)
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
    serving = find_serving_oru(UE(id=ue_id, x=x, y=y), orus)
    demand_mbps = np.random.choice([0.5, 1.0, 1.5])
    return UE(
        id=ue_id,
        x=x,
        y=y,
        connected_oru=serving.id,
        demand_mbps=demand_mbps,
    )
