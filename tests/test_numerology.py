from src.numerology import compute_num_prbs, prb_bandwidth_khz


def test_numerology_0():
    assert compute_num_prbs(10.0, 0.25, 0) == 52


def test_numerology_1():
    assert compute_num_prbs(10.0, 0.25, 1) == 26


def test_numerology_2():
    assert compute_num_prbs(10.0, 0.25, 2) == 13


def test_numerology_4():
    assert compute_num_prbs(10.0, 0.25, 4) == 3


def test_prb_bandwidth():
    assert prb_bandwidth_khz(0) == 180.0
    assert prb_bandwidth_khz(1) == 360.0
    assert prb_bandwidth_khz(2) == 720.0
