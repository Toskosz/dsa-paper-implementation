
from src.models import ORU, ORUType, UE
from src.propagation import (
    achievable_rate_mbps,
    dbm_to_watts,
    db_to_linear,
    linear_to_db,
    path_loss_db,
    received_power_dbm,
    sinr_with_sharing,
    update_ewma,
    find_serving_oru,
)


def test_path_loss_increases_with_distance():
    pl_near = path_loss_db(10.0, 2.7)
    pl_far = path_loss_db(100.0, 2.7)
    assert pl_far > pl_near


def test_path_loss_higher_exponent():
    pl_low = path_loss_db(100.0, 2.0)
    pl_high = path_loss_db(100.0, 3.0)
    assert pl_high > pl_low


def test_received_power():
    rx = received_power_dbm(46.0, 100.0, 2.7)
    assert rx < 46.0


def test_dbm_watts_roundtrip():
    assert abs(dbm_to_watts(30.0) - 1.0) < 1e-9


def test_db_linear_roundtrip():
    assert abs(linear_to_db(db_to_linear(30.0)) - 30.0) < 1e-9


def test_achievable_rate_positive():
    rate = achievable_rate_mbps(10.0, 180.0)
    assert rate > 0


def test_achievable_rate_zero_sinr():
    rate = achievable_rate_mbps(0.0, 180.0)
    assert rate == 0.0


def test_sinr_with_sharing():
    oru = ORU(id=0, x=0, y=0, coverage_radius=300, path_loss_exponent=2.7, tx_power_dbm=46, oru_type=ORUType.MACRO)
    ue = UE(id=0, x=50, y=0, connected_oru=0)
    sinr = sinr_with_sharing(ue, oru, [oru], -100.0)
    assert sinr > 0


def test_ewma_update():
    ue = UE(id=0, x=0, y=0, ewma_throughput=1.0, instant_rate=2.0)
    update_ewma(ue, alpha=0.1)
    expected = 0.9 * 1.0 + 0.1 * 2.0
    assert abs(ue.ewma_throughput - expected) < 1e-9


def test_find_serving_oru():
    orus = [
        ORU(id=0, x=0, y=0, coverage_radius=300, path_loss_exponent=2.7, tx_power_dbm=46, oru_type=ORUType.MACRO),
        ORU(id=1, x=200, y=0, coverage_radius=50, path_loss_exponent=2.8, tx_power_dbm=30, oru_type=ORUType.MICRO),
    ]
    ue = UE(id=0, x=10, y=0)
    serving = find_serving_oru(ue, orus)
    assert serving.id == 0
