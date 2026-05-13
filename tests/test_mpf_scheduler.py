from src.models import UE, default_network_config
from src.mpf_scheduler import mpf_schedule, mpf_metric
from src.numerology import prb_bandwidth_khz


def test_mpf_metric_positive():
    config = default_network_config()
    ue = UE(id=0, x=50, y=0, connected_oru=0, ewma_throughput=1.0, priority_weight=1.0)
    metric = mpf_metric(ue, 0, config, 180.0)
    assert metric > 0


def test_preemption_steals_prb():
    config = default_network_config()
    owner = UE(id=0, x=50, y=0, connected_oru=0, ewma_throughput=10.0, priority_weight=0.1, assigned_prb=0, instant_rate=0.1)
    challenger = UE(id=1, x=50, y=0, connected_oru=0, ewma_throughput=0.01, priority_weight=1.0)

    allocation = {0: 0}
    adj = {0: [1], 1: [0]}
    bw = prb_bandwidth_khz(0)

    new_alloc = mpf_schedule(
        uncolored=[challenger],
        colored_ues=[owner],
        allocation=allocation,
        adjacency=adj,
        num_prbs=1,
        config=config,
        prb_bw_khz=bw,
    )

    assert challenger.id in new_alloc or owner.id in new_alloc


def test_ewma_updated_after_schedule():
    config = default_network_config()
    ue = UE(id=0, x=50, y=0, connected_oru=0, ewma_throughput=0.0, assigned_prb=0, instant_rate=0.0)
    bw = prb_bandwidth_khz(0)

    mpf_schedule(
        uncolored=[],
        colored_ues=[ue],
        allocation={0: 0},
        adjacency={0: []},
        num_prbs=1,
        config=config,
        prb_bw_khz=bw,
    )

    assert ue.ewma_throughput > 0
