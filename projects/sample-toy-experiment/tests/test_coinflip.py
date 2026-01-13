from experiments.coinflip import Config, run_once

def test_coinflip_reasonable_estimate():
    res = run_once(Config(p_true=0.6, n=10000, seed=0))
    assert abs(res["p_hat"] - 0.6) < 0.03
