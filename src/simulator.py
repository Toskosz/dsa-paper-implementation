"""Three-phase DSA simulation driver.

Runs the complete O-RAN Dynamic Spectrum Allocation experiment in three
distinct phases, each designed to exercise a different complexity regime
of the graph coloring algorithm:

  Phase 1 — Average case O(U log U + E):
      UEs are randomly distributed across all O-RU coverage areas with
      realistic mobility (2D random walk).  The conflict graph is sparse
      to moderately dense, representing typical network conditions.

  Phase 2 — Worst case O(U²):
      All UEs are colocated at the origin (served by the Macro O-RU).
      This creates a near-complete conflict graph, maximizing the number
      of edges and stressing the coloring algorithm's inner loops.

  Phase 3 — Best case Ω(U log U):
      UEs are placed on a regular grid with 700 m spacing, ensuring
      minimal overlap.  The conflict graph has very few edges (possibly
      zero), so coloring cost is dominated by the initial sort.

Results are logged to CSV for post-hoc analysis and plotting.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from src.graph_coloring import policy_guided_coloring
from src.interference_graph import build_conflict_graph, weighted_degrees
from src.mobility import (
    random_walk_step,
    reflect_in_coverage,
    reassociate,
    spawn_ue_at_position,
    spawn_ue_in_coverage,
)
from src.models import SimulationResult, UE, default_network_config
from src.mpf_scheduler import mpf_schedule
from src.numerology import compute_num_prbs, prb_bandwidth_khz


class Simulator:
    """Main simulation controller.

    Orchestrates the three-phase experiment, managing UE creation,
    mobility, conflict graph construction, graph coloring, MPF
    scheduling, and metric computation at each time step.

    Args:
        numerology: 3GPP numerology index μ (0–4).
        phase1_steps: Number of time steps for the average-case phase.
        phase2_steps: Number of time steps for the worst-case phase.
        phase3_steps: Number of time steps for the best-case phase.
        max_ues_phase1: Upper bound on UE count in Phase 1 (random per step).
        max_ues_phase2: Maximum UE count in Phases 2 and 3.
        csv_path: Output file path for the CSV log.
    """

    def __init__(
        self,
        numerology: int = 0,
        phase1_steps: int = 450,
        phase2_steps: int = 300,
        phase3_steps: int = 150,
        max_ues_phase1: int = 160,
        max_ues_phase2: int = 160,
        csv_path: str = "results/simulation_log.csv",
    ):
        self.config = default_network_config()
        self.config.numerology = numerology
        self.num_prbs = compute_num_prbs(
            self.config.bandwidth_mhz, self.config.guard_band_mhz, numerology
        )
        self.prb_bw_khz = prb_bandwidth_khz(numerology)
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.phase3_steps = phase3_steps
        self.max_ues_phase1 = max_ues_phase1
        self.max_ues_phase2 = max_ues_phase2
        self.csv_path = csv_path
        self.ues: list[UE] = []
        self.next_ue_id = 0
        self.results: list[SimulationResult] = []
        self._t = 0

    def run(self) -> list[SimulationResult]:
        """Execute all three simulation phases sequentially.

        Returns:
            List of SimulationResult records, one per time step.
        """
        self._ensure_dir()
        self._write_csv_header()
        self._run_phase1()
        self._run_phase2()
        self._run_phase3()
        return self.results

    def _ensure_dir(self) -> None:
        """Create the output directory for the CSV log if it does not exist."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def _write_csv_header(self) -> None:
        """Write the CSV header row."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "phase", "num_ues", "num_edges",
                "exec_time_ms", "prb_success_rate", "fairness",
            ])

    def _append_result(self, result: SimulationResult) -> None:
        """Append a simulation result to the in-memory list and CSV file."""
        self.results.append(result)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                result.timestamp, result.phase, result.num_ues,
                result.num_edges, result.exec_time_ms,
                result.prb_success_rate, result.fairness,
            ])

    def _compute_metrics(self, ues: list[UE], allocation: dict) -> tuple[float, float]:
        """Compute PRB success rate and Jain's fairness index.

        Jain's fairness index over served UEs:
            J = (Σ rᵢ)² / (n · Σ rᵢ²)
        where rᵢ = min(Rᵢ / Dᵢ, 1) is the served-to-demanded rate ratio
        capped at 1 (no bonus for exceeding demand).

        Args:
            ues: All active UEs.
            allocation: Current PRB allocation dict.

        Returns:
            Tuple (prb_success_rate, fairness).
        """
        total = len(ues)
        if total == 0:
            return 0.0, 0.0
        colored = len(allocation)
        prb_success = colored / total

        served_rates = [ue.instant_rate for ue in ues if ue.assigned_prb is not None]
        if len(served_rates) == 0:
            return prb_success, 0.0

        demands = [ue.demand_mbps for ue in ues if ue.assigned_prb is not None]
        if len(demands) == 0:
            return prb_success, 0.0

        # Rate-to-demand ratios, capped at 1.0
        ratios = [min(r / d, 1.0) for r, d in zip(served_rates, demands) if d > 0]
        jain_numerator = sum(ratios) ** 2
        jain_denominator = len(ratios) * sum(r ** 2 for r in ratios) if any(ratios) else 1.0
        fairness = jain_numerator / jain_denominator if jain_denominator > 0 else 0.0

        return prb_success, fairness

    def _run_one_step(self, phase: str) -> None:
        """Execute a single scheduling time step.

        Pipeline: build conflict graph → compute weighted degrees → sort
        UEs → run graph coloring → run MPF scheduling for uncolored UEs
        → compute and log metrics.
        """
        adjacency, edge_count = build_conflict_graph(self.ues, self.config, self.prb_bw_khz)
        degrees = weighted_degrees(self.ues, adjacency)
        # Welsh-Powell ordering: highest weighted degree first
        sorted_ues = sorted(self.ues, key=lambda u: degrees.get(u.id, 0), reverse=True)

        allocation, uncolored, exec_time = policy_guided_coloring(
            sorted_ues, adjacency, self.num_prbs
        )

        # Fallback: MPF scheduling for UEs that could not be colored
        colored_ues = [u for u in self.ues if u.assigned_prb is not None]
        if uncolored:
            allocation = mpf_schedule(
                uncolored, colored_ues, allocation, adjacency,
                self.num_prbs, self.config, self.prb_bw_khz,
            )

        prb_success, fairness = self._compute_metrics(self.ues, allocation)
        self._append_result(SimulationResult(
            timestamp=self._t,
            phase=phase,
            num_ues=len(self.ues),
            num_edges=edge_count,
            exec_time_ms=exec_time,
            prb_success_rate=prb_success,
            fairness=fairness,
        ))
        self._t += 1

    def _run_phase1(self) -> None:
        """Phase 1: Average case — random UE positions with mobility.

        UEs are spawned inside random O-RU coverage areas and move via
        2D random walk.  The UE count varies randomly each step (10 to
        max_ues_phase1), producing diverse conflict graph densities.
        """
        self.ues = []
        self.next_ue_id = 0

        for step in range(self.phase1_steps):
            n_ues = np.random.randint(10, self.max_ues_phase1 + 1)
            while len(self.ues) < n_ues:
                ue = spawn_ue_in_coverage(self.next_ue_id, self.config.orus)
                self.ues.append(ue)
                self.next_ue_id += 1
            while len(self.ues) > n_ues:
                self.ues.pop(np.random.randint(len(self.ues)))

            for ue in self.ues:
                random_walk_step(ue, self.config.ue_speed_mps, self.config.scheduling_interval_s)
                reflect_in_coverage(ue, self.config.orus)
                reassociate(ue, self.config.orus)
                # Reset PRB assignment for this time step
                ue.assigned_prb = None

            self._run_one_step("average")

    def _run_phase2(self) -> None:
        """Phase 2: Worst case — colocated UEs at the origin.

        All UEs are spawned at (0, 0) and connected to the Macro O-RU,
        creating a near-complete conflict graph that forces O(U²) coloring.
        The UE count ramps linearly from 10 to max_ues_phase2.
        """
        self.ues = []
        self.next_ue_id = 0

        # Linear ramp: 10 → max_ues_phase2 over phase2_steps
        ramp = np.linspace(10, self.max_ues_phase2, self.phase2_steps, dtype=int)

        for step in range(self.phase2_steps):
            n_ues = int(ramp[step])
            while len(self.ues) < n_ues:
                ue = spawn_ue_at_position(
                    self.next_ue_id, 0.0, 0.0, self.config.orus
                )
                self.ues.append(ue)
                self.next_ue_id += 1

            for ue in self.ues:
                ue.assigned_prb = None

            self._run_one_step("worst")

    def _run_phase3(self) -> None:
        """Phase 3: Best case — widely dispersed UEs on a grid.

        UEs are placed on a regular √n × √n grid with 700 m spacing,
        ensuring minimal inter-UE interference (few or zero edges).
        Coloring cost is dominated by the O(U log U) sort.
        """
        self.ues = []
        self.next_ue_id = 0

        n_ues = self.max_ues_phase2
        spacing = 700.0
        # Build a square grid large enough to hold all UEs
        side = int(np.ceil(np.sqrt(n_ues)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= n_ues:
                    break
                x = i * spacing
                y = j * spacing
                positions.append((x, y))

        for idx in range(n_ues):
            x, y = positions[idx]
            oru = self.config.orus[0]
            ue = UE(
                id=self.next_ue_id,
                x=x,
                y=y,
                connected_oru=oru.id,
                demand_mbps=np.random.choice([0.5, 1.0, 1.5]),
            )
            self.ues.append(ue)
            self.next_ue_id += 1

        for step in range(self.phase3_steps):
            for ue in self.ues:
                ue.assigned_prb = None
            self._run_one_step("best")
