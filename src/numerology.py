"""3GPP numerology utilities for PRB computation.

Implements the formula from the paper to determine how many Physical
Resource Blocks (PRBs) are available given a channel bandwidth,
guard band, and numerology index μ.

PRB count:  P = ⌊(B − 2·B_G) / (Δf · 12)⌋
where Δf = 15 kHz · 2^μ is the subcarrier spacing.
Each PRB spans 12 consecutive subcarriers in frequency.
"""

from __future__ import annotations

import numpy as np


def compute_num_prbs(
    bandwidth_mhz: float = 10.0,
    guard_band_mhz: float = 0.25,
    numerology: int = 0,
) -> int:
    """Compute the number of usable PRBs for a given numerology.

    Args:
        bandwidth_mhz: Total channel bandwidth B in MHz.
        guard_band_mhz: Guard band B_G on each side in MHz.
        numerology: 3GPP numerology index μ (0–4).

    Returns:
        Number of PRBs (at least 1).
    """
    usable_bw_hz = (bandwidth_mhz - 2 * guard_band_mhz) * 1e6
    subcarrier_spacing_hz = 15e3 * (2**numerology)
    # Each PRB occupies 12 subcarriers in frequency
    prbs = int(np.floor(usable_bw_hz / (subcarrier_spacing_hz * 12)))
    return max(prbs, 1)


def prb_bandwidth_khz(numerology: int) -> float:
    """Return the bandwidth of a single PRB in kHz.

    A PRB spans 12 subcarriers, each separated by Δf = 15·2^μ kHz.
    """
    subcarrier_spacing_hz = 15e3 * (2**numerology)
    return subcarrier_spacing_hz * 12 / 1e3
