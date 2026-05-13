from __future__ import annotations

import numpy as np

from src.models import ORU, UE

REFERENCE_DISTANCE_M = 1.0
PATH_LOSS_REF_DB = 30.0


def path_loss_db(distance_m: float, exponent: float) -> float:
    if distance_m < REFERENCE_DISTANCE_M:
        distance_m = REFERENCE_DISTANCE_M
    return PATH_LOSS_REF_DB + 10 * exponent * np.log10(distance_m)


def received_power_dbm(tx_power_dbm: float, distance_m: float, exponent: float) -> float:
    return tx_power_dbm - path_loss_db(distance_m, exponent)


def dbm_to_watts(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0) / 1e3


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    if linear <= 0:
        return -np.inf
    return 10.0 * np.log10(linear)


def sinr_linear(ue: UE, serving_oru: ORU, interfering_orus: list[ORU]) -> float:
    dist_s = np.sqrt((ue.x - serving_oru.x) ** 2 + (ue.y - serving_oru.y) ** 2)
    p_signal_w = dbm_to_watts(
        received_power_dbm(serving_oru.tx_power_dbm, dist_s, serving_oru.path_loss_exponent)
    )
    interference_w = 0.0
    for oru in interfering_orus:
        if oru.id == serving_oru.id:
            continue
        dist_i = np.sqrt((ue.x - oru.x) ** 2 + (ue.y - oru.y) ** 2)
        interference_w += dbm_to_watts(
            received_power_dbm(oru.tx_power_dbm, dist_i, oru.path_loss_exponent)
        )
    return p_signal_w / interference_w


def sinr_with_sharing(
    ue: UE,
    serving_oru: ORU,
    interfering_orus: list[ORU],
    noise_power_dbm: float,
    sharing_penalty_db: float = 0.0,
) -> float:
    dist_s = np.sqrt((ue.x - serving_oru.x) ** 2 + (ue.y - serving_oru.y) ** 2)
    p_signal_dbm = received_power_dbm(serving_oru.tx_power_dbm, dist_s, serving_oru.path_loss_exponent)
    p_signal_dbm -= sharing_penalty_db
    p_signal_w = dbm_to_watts(p_signal_dbm)

    interference_w = 0.0
    for oru in interfering_orus:
        if oru.id == serving_oru.id:
            continue
        dist_i = np.sqrt((ue.x - oru.x) ** 2 + (ue.y - oru.y) ** 2)
        interference_w += dbm_to_watts(
            received_power_dbm(oru.tx_power_dbm, dist_i, oru.path_loss_exponent)
        )

    noise_w = dbm_to_watts(noise_power_dbm)
    return p_signal_w / (interference_w + noise_w)


def achievable_rate_mbps(sinr_lin: float, prb_bw_khz: float) -> float:
    bw_hz = prb_bw_khz * 1e3
    if sinr_lin <= 0:
        return 0.0
    return bw_hz * np.log2(1.0 + sinr_lin) / 1e6


def find_serving_oru(ue: UE, orus: list[ORU]) -> ORU:
    best_oru = orus[0]
    best_rx = -np.inf
    for oru in orus:
        dist = np.sqrt((ue.x - oru.x) ** 2 + (ue.y - oru.y) ** 2)
        if dist > oru.coverage_radius:
            continue
        rx = received_power_dbm(oru.tx_power_dbm, dist, oru.path_loss_exponent)
        if rx > best_rx:
            best_rx = rx
            best_oru = oru
    return best_oru


def update_ewma(ue: UE, alpha: float = 0.1) -> None:
    ue.ewma_throughput = (1.0 - alpha) * ue.ewma_throughput + alpha * ue.instant_rate
