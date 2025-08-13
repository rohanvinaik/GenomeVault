from genomevault.federated.simulator import simulate_round

def test_fedavg_smoke():
    res = simulate_round(n_clients=3, dim=8)
    assert hasattr(res, "aggregated") or isinstance(res, (list, tuple, dict))
