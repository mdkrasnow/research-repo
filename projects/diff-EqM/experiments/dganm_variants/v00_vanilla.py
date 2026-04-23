"""v00_vanilla — sanity baseline.

Plain EqM loss, no mining. Must match `train_cifar_eqm_unet.py` on FID
(within seed noise). If this doesn't, the shared harness has a bug and
no DG-ANM variant result is interpretable.
"""

from ._common import TrainArgs, eqm_loss, train_loop


def step_fn(model, x, step, device, args: TrainArgs):
    loss = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    return loss, {"base": loss.item()}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base"])
