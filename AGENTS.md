# AGENTS.md

## Project

Implementation of the DSA (Dynamic Spectrum Allocation) framework from [arXiv:2601.13769](https://arxiv.org/abs/2601.13769) — an interoperable rApp/xApp architecture over O-RAN using graph-theoretic PRB assignment and modified proportional fair scheduling.

## Status

Core implementation complete. All modules implemented, 26 tests passing, lint clean. Simulation produces CSV logs and matplotlib plots for PCA complexity analysis.

## Commands

```bash
# Run tests
python3 -m pytest tests/ -v

# Lint
python3 -m ruff check src/ tests/ experiments/

# Run simulation (full)
python3 -m experiments.run_experiment

# Run simulation (quick smoke test)
python3 -m experiments.run_experiment --phase1-steps 20 --phase2-steps 10 --phase3-steps 10 --max-ues-phase1 50 --max-ues-phase2 200

# Generate plots from CSV
python3 -m experiments.plot_results --csv results/simulation_log.csv
```

## Project Structure

```
src/
  models.py              # Dataclasses: UE, ORU, NetworkConfig, SimulationResult
  numerology.py          # PRB count from μ/B/Bg formula
  propagation.py         # Path-loss, SINR, Shannon rate, EWMA throughput
  interference_graph.py  # Conflict graph G(t) construction
  graph_coloring.py      # Algorithm 1: Policy-Guided Graph Coloring
  mpf_scheduler.py       # Algorithm 2: MPF Scheduling with preemption
  mobility.py            # 2D random walk, coverage reflection, ORU reassociation
  simulator.py           # 3-phase simulator (average/worst/best case)
experiments/
  run_experiment.py      # CLI entry point
  plot_results.py        # 5 matplotlib plots for complexity analysis
tests/
  test_numerology.py
  test_propagation.py
  test_interference_graph.py
  test_graph_coloring.py
  test_mpf_scheduler.py
results/                 # CSV logs and PNG plots (gitignored)
```

## Key Domain Concepts

- **rApp**: Runs on Non-RT RIC; predicts traffic and generates spectrum policies (minutes timescale). Not modeled in this simulation — focus is xApp only.
- **xApp**: Runs on Near-RT RIC; builds a user-centric conflict graph and performs fairness-aware PRB allocation (sub-second timescale)
- **MPF scheduling**: Conflict-aware Modified Proportional Fair mechanism to avoid user starvation
- **Numerology**: PRB count is dynamic via $P = \lfloor(B - 2B_G)/(180 \cdot 2^\mu)\rfloor$ with B=10 MHz, B_G=0.25 MHz, μ=0..4
- **Three-phase simulation**: Average case O(U log U + E), worst case O(U²) via colocated UEs, best case Ω(U log U) via dispersed UEs
- Paper targets PRB assignment success rate >90% and service-share fairness >85%

## Network Parameters (from paper)

- 3 O-RUs: 1 Macro at (0,0) radius 300m, 2 Micro at (±200,0) radius 50m
- Path-loss exponents: Macro α=2.7, Micro α=2.8
- UE speed: 1.5 m/s random walk
- UE demands: 0.5 / 1.0 / 1.5 Mbps (random)
- EWMA smoothing: α=0.1
- Phase 2 scales to 5000 UEs at (0,0) for worst-case complete graph
