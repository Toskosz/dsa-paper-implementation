"""Data models for the DSA simulation.

Defines the core entities — O-RU base stations, User Equipment (UE),
network configuration parameters, and simulation result records — used
throughout the O-RAN Dynamic Spectrum Allocation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ORUType(Enum):
    """Radio Unit type: Macro cells have larger coverage and higher power; Micro cells are smaller."""
    MACRO = "macro"
    MICRO = "micro"


@dataclass
class ORU:
    """O-RAN Radio Unit (base station).

    Attributes:
        id: Unique identifier for this RU.
        x, y: Cartesian position in meters.
        coverage_radius: Maximum service distance in meters.
        path_loss_exponent: Propagation exponent α (higher ⇒ faster signal decay).
        tx_power_dbm: Per-PRB transmit power in dBm.
        oru_type: MACRO or MICRO — determines path-loss and power behavior.
    """
    id: int
    x: float
    y: float
    coverage_radius: float
    path_loss_exponent: float
    tx_power_dbm: float
    oru_type: ORUType


@dataclass
class UE:
    """User Equipment (mobile device).

    Tracks position, mobility state, network association, traffic demand,
    and scheduling state (PRB assignment, throughput metrics).

    Attributes:
        id: Unique identifier.
        x, y: Current position in meters.
        vx, vy: Current velocity components in m/s.
        connected_oru: ID of the serving O-RU (None if out of coverage).
        demand_mbps: Requested data rate in Mbps (0.5 / 1.0 / 1.5).
        priority_weight: Scheduling weight w_u (default 1.0).
        satisfaction_tolerance: Fraction of demand that is acceptable (0 = strict).
        ewma_throughput: Exponentially-weighted moving average throughput (Mbps).
        assigned_prb: PRB index assigned by the coloring/scheduling algorithm.
        instant_rate: Instantaneous achievable rate on the assigned PRB (Mbps).
    """
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
        """Return the UE's position as a 2D numpy vector."""
        return np.array([self.x, self.y])


@dataclass
class NetworkConfig:
    """Global network simulation parameters.

    Attributes:
        orus: List of O-RU base stations in the topology.
        bandwidth_mhz: Total channel bandwidth (MHz). Default 10 MHz per paper.
        guard_band_mhz: Guard band on each side of the spectrum (MHz).
        numerology: 3GPP numerology index μ (0–4); controls subcarrier spacing.
        noise_psd_dbm_hz: Thermal noise power spectral density (dBm/Hz).
        min_prb_power_dbm: Minimum per-PRB transmit power (dBm).
        ue_speed_mps: Constant UE movement speed (m/s).
        ewma_alpha: Smoothing factor for throughput averaging (0 < α ≤ 1).
        scheduling_interval_s: Duration of one scheduling slot in seconds.
    """
    orus: list[ORU] = field(default_factory=list)
    bandwidth_mhz: float = 10.0
    guard_band_mhz: float = 0.25
    numerology: int = 0
    noise_psd_dbm_hz: float = -174.0
    min_prb_power_dbm: float = 0.0
    ue_speed_mps: float = 1.5
    ewma_alpha: float = 0.1
    scheduling_interval_s: float = 1.0

    @property
    def noise_per_prb_dbm(self) -> float:
        """Thermal noise power per PRB (dBm).

        Computed as N₀ + 10·log₁₀(PRB_BW), where PRB bandwidth
        equals 12 subcarriers × (15 kHz · 2^μ).
        """
        prb_bw_hz = 15e3 * (2 ** self.numerology) * 12
        return self.noise_psd_dbm_hz + 10 * np.log10(prb_bw_hz)

    @property
    def num_prbs(self) -> int:
        """Number of usable PRBs from the 3GPP numerology formula.

        P = ⌊(B − 2·B_G) / (180 · 2^μ)⌋, where usable bandwidth is
        total bandwidth minus two guard bands, and each PRB occupies
        180 kHz × 2^μ of spectrum.
        """
        usable_bw = self.bandwidth_mhz - 2 * self.guard_band_mhz
        return int(np.floor(usable_bw * 1e6 / (180 * 2**self.numerology) / 1e3))


@dataclass
class SimulationResult:
    """Metrics recorded for a single scheduling time step.

    Attributes:
        timestamp: Step index within the simulation.
        phase: Simulation phase label ("average", "worst", or "best").
        num_ues: Total number of active UEs at this step.
        num_edges: Number of conflict edges in the interference graph.
        exec_time_ms: Wall-clock execution time of the coloring algorithm (ms).
        prb_success_rate: Fraction of UEs that received a PRB assignment.
        fairness: Jain's fairness index over served users' rate/demand ratios.
    """
    timestamp: int
    phase: str
    num_ues: int
    num_edges: int
    exec_time_ms: float
    prb_success_rate: float
    fairness: float


def default_network_config() -> NetworkConfig:
    """Create the paper's reference topology: 1 Macro + 2 Micro O-RUs.

    - Macro O-RU at the origin with 300 m radius, α=2.7, 20 dBm.
    - Two Micro O-RUs at (±200, 0) with 50 m radius, α=2.8, 10 dBm.
    """
    orus = [
        ORU(
            id=0,
            x=0.0,
            y=0.0,
            coverage_radius=300.0,
            path_loss_exponent=2.7,
            tx_power_dbm=20.0,
            oru_type=ORUType.MACRO,
        ),
        ORU(
            id=1,
            x=200.0,
            y=0.0,
            coverage_radius=50.0,
            path_loss_exponent=2.8,
            tx_power_dbm=10.0,
            oru_type=ORUType.MICRO,
        ),
        ORU(
            id=2,
            x=-200.0,
            y=0.0,
            coverage_radius=50.0,
            path_loss_exponent=2.8,
            tx_power_dbm=10.0,
            oru_type=ORUType.MICRO,
        ),
    ]
    return NetworkConfig(orus=orus)
