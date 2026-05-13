from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ORUType(Enum):
    MACRO = "macro"
    MICRO = "micro"


@dataclass
class ORU:
    id: int
    x: float
    y: float
    coverage_radius: float
    path_loss_exponent: float
    tx_power_dbm: float
    oru_type: ORUType


@dataclass
class UE:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    connected_oru: Optional[int] = None
    demand_mbps: float = 1.0
    priority_weight: float = 1.0
    satisfaction_tolerance: float = 0.0
    ewma_throughput: float = 0.0
    assigned_prb: Optional[int] = None
    instant_rate: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class NetworkConfig:
    orus: list[ORU] = field(default_factory=list)
    bandwidth_mhz: float = 10.0
    guard_band_mhz: float = 0.25
    numerology: int = 0
    noise_power_dbm: float = -100.0
    ue_speed_mps: float = 1.5
    ewma_alpha: float = 0.1
    scheduling_interval_s: float = 1.0

    @property
    def num_prbs(self) -> int:
        usable_bw = self.bandwidth_mhz - 2 * self.guard_band_mhz
        return int(np.floor(usable_bw * 1e6 / (180 * 2**self.numerology) / 1e3))


@dataclass
class SimulationResult:
    timestamp: int
    phase: str
    num_ues: int
    num_edges: int
    exec_time_ms: float
    prb_success_rate: float
    fairness: float


def default_network_config() -> NetworkConfig:
    orus = [
        ORU(
            id=0,
            x=0.0,
            y=0.0,
            coverage_radius=300.0,
            path_loss_exponent=2.7,
            tx_power_dbm=46.0,
            oru_type=ORUType.MACRO,
        ),
        ORU(
            id=1,
            x=200.0,
            y=0.0,
            coverage_radius=50.0,
            path_loss_exponent=2.8,
            tx_power_dbm=30.0,
            oru_type=ORUType.MICRO,
        ),
        ORU(
            id=2,
            x=-200.0,
            y=0.0,
            coverage_radius=50.0,
            path_loss_exponent=2.8,
            tx_power_dbm=30.0,
            oru_type=ORUType.MICRO,
        ),
    ]
    return NetworkConfig(orus=orus)
