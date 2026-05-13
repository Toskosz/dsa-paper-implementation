from src.graph_coloring import policy_guided_coloring
from src.models import UE


def test_no_edges_all_colored():
    ues = [UE(id=i, x=0, y=0) for i in range(5)]
    adj = {i: [] for i in range(5)}
    alloc, uncolored, _ = policy_guided_coloring(ues, adj, num_prbs=10)
    assert len(alloc) == 5
    assert len(uncolored) == 0


def test_complete_graph_limited_prbs():
    n = 5
    ues = [UE(id=i, x=0, y=0) for i in range(n)]
    adj = {i: [j for j in range(n) if j != i] for i in range(n)}
    alloc, uncolored, _ = policy_guided_coloring(ues, adj, num_prbs=3)
    assert len(alloc) == 3
    assert len(uncolored) == 2


def test_partial_conflicts():
    ues = [UE(id=i, x=0, y=0) for i in range(4)]
    adj = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }
    alloc, uncolored, _ = policy_guided_coloring(ues, adj, num_prbs=3)
    assert len(uncolored) == 0
    assert alloc[0] != alloc[1]
    assert alloc[1] != alloc[2]
    assert alloc[2] != alloc[3]


def test_no_prbs_all_uncolored():
    ues = [UE(id=i, x=0, y=0) for i in range(3)]
    adj = {i: [] for i in range(3)}
    alloc, uncolored, _ = policy_guided_coloring(ues, adj, num_prbs=0)
    assert len(alloc) == 0
    assert len(uncolored) == 3
