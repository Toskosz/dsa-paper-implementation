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
        self._ensure_dir()
        self._write_csv_header()
        self._run_phase1()
        self._run_phase2()
        self._run_phase3()
        return self.results

    def _ensure_dir(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def _write_csv_header(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "phase", "num_ues", "num_edges",
                "exec_time_ms", "prb_success_rate", "fairness",
            ])

    def _append_result(self, result: SimulationResult) -> None:
        self.results.append(result)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                result.timestamp, result.phase, result.num_ues,
                result.num_edges, result.exec_time_ms,
                result.prb_success_rate, result.fairness,
            ])

    def _compute_metrics(self, ues: list[UE], allocation: dict) -> tuple[float, float]:
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

        ratios = [min(r / d, 1.0) for r, d in zip(served_rates, demands) if d > 0]
        jain_numerator = sum(ratios) ** 2
        jain_denominator = len(ratios) * sum(r ** 2 for r in ratios) if any(ratios) else 1.0
        fairness = jain_numerator / jain_denominator if jain_denominator > 0 else 0.0

        return prb_success, fairness

    def _run_one_step(self, phase: str) -> None:
        adjacency, edge_count = build_conflict_graph(self.ues, self.config, self.prb_bw_khz)
        degrees = weighted_degrees(self.ues, adjacency)
        sorted_ues = sorted(self.ues, key=lambda u: degrees.get(u.id, 0), reverse=True)

        allocation, uncolored, exec_time = policy_guided_coloring(
            sorted_ues, adjacency, self.num_prbs
        )

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
                ue.assigned_prb = None

            self._run_one_step("average")

    def _run_phase2(self) -> None:
        self.ues = []
        self.next_ue_id = 0

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
        self.ues = []
        self.next_ue_id = 0

        n_ues = self.max_ues_phase2
        spacing = 700.0
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
