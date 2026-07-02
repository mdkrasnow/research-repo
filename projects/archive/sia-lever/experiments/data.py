"""
Data generator for the rotation + shortcut-channel symmetry toy.

Task (clean):
    x     = r * [cos(theta), sin(theta)]
    target= r * [cos(theta + delta), sin(theta + delta)]   (rotate x by delta)

Model input is a 6-vector:
    [x_x, x_y, sin(delta), cos(delta), shortcut_x, shortcut_y]

Shortcut channel leaks the TARGET direction:
    clean batch:    shortcut = [cos(theta+delta), sin(theta+delta)]   (== target dir, scaled to unit)
A model can hit low MSE by reading the shortcut and ignoring the rotation structure.

Negative control (broken symmetry):
    target_fake = r * [cos(psi), sin(psi)]   with psi random, UNRELATED to theta+delta
    shortcut    = [cos(psi), sin(psi)]
An honest rotation-learner CANNOT solve this (no rotation maps x->target_fake consistently),
so it must FAIL. A shortcut-reader still succeeds. High neg-control MSE == honest. That is the tell.

shortcut_randomized: same clean task but shortcut channel filled with random unit vectors.
Used to measure shortcut_sensitivity (MSE jump => model was leaning on the shortcut).
"""

import torch


def _unit(angle):
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def make_batch(n, mode="clean", seed=None, r_lo=0.5, r_hi=1.5, leak_alpha=1.0):
    """Return dict of tensors for one batch.

    mode:
      "clean"       -> honest rotation task, shortcut leaks true target
      "neg_control" -> broken-symmetry task, shortcut leaks fake target
      "shortcut_rand" -> clean task but shortcut channel is random (probe)

    leak_alpha in [0,1]: strength of the shortcut leak. 1.0 = full target leaked (default,
      adversarial). <1.0 = partial/noisy leak:  shortcut = alpha*target + (1-alpha)*noise.
      Used by the leak-strength robustness sweep to show the phenomenon is not an artifact of
      a perfectly clean leak.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    theta = (torch.rand(n, generator=g) * 2 - 1) * torch.pi      # [-pi, pi]
    delta = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
    r = r_lo + torch.rand(n, generator=g) * (r_hi - r_lo)

    x = r.unsqueeze(-1) * _unit(theta)                            # [n,2]
    target_dir = _unit(theta + delta)                            # honest target direction

    def _leak(target_vec):
        # partial leak: blend the leaked target with random noise of matched scale
        if leak_alpha >= 1.0:
            return target_vec.clone()
        phi = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
        rr = r_lo + torch.rand(n, generator=g) * (r_hi - r_lo)
        noise = rr.unsqueeze(-1) * _unit(phi)
        return leak_alpha * target_vec + (1.0 - leak_alpha) * noise

    # Shortcut channel leaks the (full or partial) target vector y. A prediction-only learner can
    # copy the shortcut and score low MSE without learning rotation at all.
    if mode == "neg_control":
        psi = (torch.rand(n, generator=g) * 2 - 1) * torch.pi    # unrelated angle
        y = r.unsqueeze(-1) * _unit(psi)
        shortcut = _leak(y)                                      # leaks the FAKE target => cheater wins broken task
    else:
        y = r.unsqueeze(-1) * target_dir
        if mode == "shortcut_rand":
            phi = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
            r2 = r_lo + torch.rand(n, generator=g) * (r_hi - r_lo)
            shortcut = r2.unsqueeze(-1) * _unit(phi)             # random vector, no info
        else:  # clean
            shortcut = _leak(y)                                  # leaks true target

    sin_d = torch.sin(delta).unsqueeze(-1)
    cos_d = torch.cos(delta).unsqueeze(-1)
    inp = torch.cat([x, sin_d, cos_d, shortcut], dim=-1)         # [n,6]

    return {
        "input": inp,
        "x": x,
        "delta": delta,
        "y": y,
        "theta": theta,
        "r": r,
    }
