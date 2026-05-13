from __future__ import annotations

import numpy as np


def compute_num_prbs(
    bandwidth_mhz: float = 10.0,
    guard_band_mhz: float = 0.25,
    numerology: int = 0,
) -> int:
    usable_bw_hz = (bandwidth_mhz - 2 * guard_band_mhz) * 1e6
    subcarrier_spacing_hz = 15e3 * (2**numerology)
    prbs = int(np.floor(usable_bw_hz / (subcarrier_spacing_hz * 12)))
    return max(prbs, 1)


def prb_bandwidth_khz(numerology: int) -> float:
    subcarrier_spacing_hz = 15e3 * (2**numerology)
    return subcarrier_spacing_hz * 12 / 1e3
