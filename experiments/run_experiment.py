from __future__ import annotations

import argparse

from src.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSA simulation experiment")
    parser.add_argument("--numerology", type=int, default=0, help="Numerology index (0-4)")
    parser.add_argument("--phase1-steps", type=int, default=450)
    parser.add_argument("--phase2-steps", type=int, default=300)
    parser.add_argument("--phase3-steps", type=int, default=150)
    parser.add_argument("--max-ues-phase1", type=int, default=160)
    parser.add_argument("--max-ues-phase2", type=int, default=160)
    parser.add_argument("--csv-path", type=str, default="results/simulation_log.csv")
    args = parser.parse_args()

    sim = Simulator(
        numerology=args.numerology,
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        phase3_steps=args.phase3_steps,
        max_ues_phase1=args.max_ues_phase1,
        max_ues_phase2=args.max_ues_phase2,
        csv_path=args.csv_path,
    )
    results = sim.run()
    print(f"Simulation complete: {len(results)} timesteps logged to {args.csv_path}")


if __name__ == "__main__":
    main()
