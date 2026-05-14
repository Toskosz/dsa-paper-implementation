from src.models import ORU, ORUType, UE, default_network_config
from src.interference_graph import build_conflict_graph, weighted_degrees


def _make_orus():
    return [
        ORU(id=0, x=0, y=0, coverage_radius=300, path_loss_exponent=2.7, tx_power_dbm=20, oru_type=ORUType.MACRO),
        ORU(id=1, x=200, y=0, coverage_radius=50, path_loss_exponent=2.8, tx_power_dbm=10, oru_type=ORUType.MICRO),
    ]


def test_same_oru_creates_edge():
    config = default_network_config()
    ues = [
        UE(id=0, x=10, y=0, connected_oru=0, demand_mbps=1.0, satisfaction_tolerance=0.0),
        UE(id=1, x=20, y=0, connected_oru=0, demand_mbps=1.0, satisfaction_tolerance=0.0),
    ]
    adj, edges = build_conflict_graph(ues, config, 180.0)
    assert edges == 1
    assert 1 in adj[0]
    assert 0 in adj[1]


def test_no_edge_far_apart():
    config = default_network_config()
    ues = [
        UE(id=0, x=10, y=0, connected_oru=0, demand_mbps=1.0, satisfaction_tolerance=0.0),
        UE(id=1, x=200, y=0, connected_oru=1, demand_mbps=1.0, satisfaction_tolerance=0.99),
    ]
    adj, edges = build_conflict_graph(ues, config, 180.0)
    assert edges == 0


def test_weighted_degrees():
    config = default_network_config()
    ues = [
        UE(id=0, x=10, y=0, connected_oru=0, priority_weight=2.0),
        UE(id=1, x=20, y=0, connected_oru=0, priority_weight=1.0),
    ]
    adj, _ = build_conflict_graph(ues, config, 180.0)
    degrees = weighted_degrees(ues, adj)
    assert degrees[0] == 2.0
    assert degrees[1] == 1.0


def test_no_ues_empty_graph():
    config = default_network_config()
    adj, edges = build_conflict_graph([], config, 180.0)
    assert edges == 0
    assert len(adj) == 0
