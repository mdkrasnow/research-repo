"""
2-process gloo (CPU) test for exact-fwrev's explicit DDP gradient allreduce
(fb_direct/exact_hvp.py::allreduce_fwrev_grads) BEFORE any multi-GPU time.

exact-fwrev bypasses loss.backward(), so DDP's hooks never fire; the manual
flat allreduce is the replacement. This test verifies, with two real
torch.distributed processes:

  1. Each rank's post-allreduce .grad equals the cross-rank AVERAGE of the
     pre-allreduce grads (DDP semantics), exactly, for every parameter.
  2. Parameters with .grad None are left untouched.
  3. fwrev_rank_sync_checksum: identical params -> zero MIN/MAX spread;
     a deliberately perturbed rank -> nonzero spread (the desync guard's
     detection actually fires).

Run: python tests/test_fb_direct_exact_hvp_ddp.py  (CPU, ~15s)
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORLD_SIZE = 2
PORT = "29611"


def _make_model(seed):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4),
    ).double()
    return m


def _worker(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = PORT
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    from fb_direct.exact_hvp import allreduce_fwrev_grads, fwrev_rank_sync_checksum

    # Same params on every rank (same seed), different grads per rank.
    model = _make_model(seed=0)
    torch.manual_seed(100 + rank)
    rank_grads = {}
    for i, p in enumerate(model.parameters()):
        if i == len(list(model.parameters())) - 1:
            p.grad = None  # last param: no grad, must be left untouched
        else:
            p.grad = torch.randn_like(p)
        rank_grads[i] = None if p.grad is None else p.grad.clone()

    # Expected average: gather all ranks' grads via the same collective set.
    expected = {}
    for i, p in enumerate(model.parameters()):
        if rank_grads[i] is None:
            expected[i] = None
            continue
        acc = rank_grads[i].clone()
        dist.all_reduce(acc)
        expected[i] = acc / WORLD_SIZE

    allreduce_fwrev_grads(model)

    for i, p in enumerate(model.parameters()):
        if expected[i] is None:
            assert p.grad is None, f"rank {rank}: param {i} grad should stay None"
        else:
            torch.testing.assert_close(p.grad, expected[i], rtol=1e-12, atol=1e-14)

    # Checksum: identical params -> zero spread.
    cs = fwrev_rank_sync_checksum(model)
    mn, mx = cs.clone(), cs.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    assert float(mx - mn) == 0.0, f"identical ranks show spread {float(mx - mn)}"

    # Perturb rank 1 -> spread must become nonzero (guard actually detects).
    if rank == 1:
        with torch.no_grad():
            next(model.parameters()).add_(1e-3)
    cs2 = fwrev_rank_sync_checksum(model)
    mn2, mx2 = cs2.clone(), cs2.clone()
    dist.all_reduce(mn2, op=dist.ReduceOp.MIN)
    dist.all_reduce(mx2, op=dist.ReduceOp.MAX)
    assert float(mx2 - mn2) > 1e-7, "desync guard failed to detect a perturbed rank"

    if rank == 0:
        print("PASS allreduce_fwrev_grads: exact cross-rank average, None grads untouched")
        print("PASS fwrev_rank_sync_checksum: zero spread when synced, detects perturbation")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(_worker, nprocs=WORLD_SIZE, join=True)
    print("ALL DDP TESTS PASSED")
