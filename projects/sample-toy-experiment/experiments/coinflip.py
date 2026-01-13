import json, random, argparse
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    p_true: float
    n: int
    seed: int

def run_once(cfg: Config) -> Dict[str, Any]:
    rng = random.Random(cfg.seed)
    xs = [1 if rng.random() < cfg.p_true else 0 for _ in range(cfg.n)]
    p_hat = sum(xs) / len(xs)
    import math
    se = math.sqrt(p_hat * (1 - p_hat) / cfg.n)
    ci = (max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se))
    return {"p_true": cfg.p_true, "n": cfg.n, "seed": cfg.seed, "p_hat": p_hat, "ci_low": ci[0], "ci_high": ci[1]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(open(args.config, "r", encoding="utf-8").read())
    cfg = Config(p_true=float(data["p_true"]), n=int(data["n"]), seed=int(data["seed"]))
    res = run_once(cfg)
    s = json.dumps(res, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(s + "\n")
    else:
        print(s)

if __name__ == "__main__":
    main()
