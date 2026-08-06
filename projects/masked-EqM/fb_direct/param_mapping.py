"""
Explicit forward<->backward parameter-mapping registry: Pi (sync
theta->phi) and Pi-dagger (transfer a phi update back into theta).

Every entry is keyed by FULLY QUALIFIED parameter name on both sides --
never by iteration order (Section 5.3 of the spec: "Never rely on matching
parameters only by iteration order").

For this architecture every reverse-active / recomputed_conditioning
weight tensor is mapped with `mapping_type="identity"` (same shape, same
orientation) -- see reverse_model.py's docstring for why (our manual VJPs
are `u_y @ W`, which reuses the forward nn.Linear.weight layout directly,
and conv_transpose2d reuses the forward Conv2d weight layout directly for
the patch embed). A `mapping_type="transpose"` implementation is included
for architectural generality / unit coverage even though nothing in this
model currently uses it.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


CATEGORY_REVERSE_ACTIVE = "reverse_active"
CATEGORY_RECOMPUTED_CONDITIONING = "recomputed_conditioning"
CATEGORY_CACHE_ONLY = "cache_only"
CATEGORY_UNUSED = "unused"

_VALID_CATEGORIES = {
    CATEGORY_REVERSE_ACTIVE,
    CATEGORY_RECOMPUTED_CONDITIONING,
    CATEGORY_CACHE_ONLY,
    CATEGORY_UNUSED,
}


@dataclass
class MappingEntry:
    forward_name: str
    backward_name: Optional[str]  # None for cache_only / unused
    mapping_type: Optional[str]  # "identity" | "transpose" | None
    category: str
    forward_shape: tuple
    backward_shape: Optional[tuple]

    def __post_init__(self):
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(f"unknown category {self.category!r} for {self.forward_name}")
        if self.category in (CATEGORY_REVERSE_ACTIVE, CATEGORY_RECOMPUTED_CONDITIONING):
            if self.backward_name is None or self.mapping_type is None:
                raise ValueError(
                    f"{self.forward_name} is {self.category} but has no backward mapping"
                )


def _tie(forward_param, backward_param, mapping_type):
    if mapping_type == "identity":
        if forward_param.shape != backward_param.shape:
            raise ValueError(
                f"identity mapping shape mismatch: {tuple(forward_param.shape)} "
                f"vs {tuple(backward_param.shape)}"
            )
        backward_param.data.copy_(forward_param.data)
    elif mapping_type == "transpose":
        expected = forward_param.shape[::-1]
        if tuple(expected) != tuple(backward_param.shape):
            raise ValueError(
                f"transpose mapping shape mismatch: {tuple(forward_param.shape)}^T != "
                f"{tuple(backward_param.shape)}"
            )
        backward_param.data.copy_(forward_param.data.t())
    else:
        raise NotImplementedError(f"mapping_type={mapping_type!r}")


def _apply_delta(forward_param, delta, mapping_type):
    if mapping_type == "identity":
        forward_param.data.add_(delta)
    elif mapping_type == "transpose":
        forward_param.data.add_(delta.t())
    else:
        raise NotImplementedError(f"mapping_type={mapping_type!r}")


def build_default_mapping(forward_model, reverse_model):
    """Construct the MappingEntry list for one EqM<->ReverseEqM pair.

    Explicit, hand-written per this exact architecture (models.py's SiTBlock
    / ScalarEnergyHead / PatchEmbed) -- not inferred generically.
    """
    fwd_params = dict(forward_model.named_parameters())
    bwd_params = dict(reverse_model.named_parameters())
    entries: List[MappingEntry] = []

    def entry(fname, bname, mtype, category):
        fshape = tuple(fwd_params[fname].shape) if fname in fwd_params else None
        bshape = tuple(bwd_params[bname].shape) if bname is not None else None
        entries.append(MappingEntry(fname, bname, mtype, category, fshape, bshape))

    entry("x_embedder.proj.weight", "x_embedder_weight", "identity", CATEGORY_REVERSE_ACTIVE)
    entry("x_embedder.proj.bias", None, None, CATEGORY_CACHE_ONLY)

    for name in fwd_params:
        if name.startswith("t_embedder.") or name.startswith("y_embedder."):
            entry(name, None, None, CATEGORY_CACHE_ONLY)
        elif name == "pos_embed":
            entry(name, None, None, CATEGORY_UNUSED)

    depth = len(forward_model.blocks)
    for i in range(depth):
        p = f"blocks.{i}."
        b = f"blocks.{i}."
        entry(p + "adaLN_modulation.1.weight", b + "adaLN_modulation.1.weight",
              "identity", CATEGORY_RECOMPUTED_CONDITIONING)
        entry(p + "adaLN_modulation.1.bias", b + "adaLN_modulation.1.bias",
              "identity", CATEGORY_RECOMPUTED_CONDITIONING)
        entry(p + "attn.qkv.weight", b + "attn_qkv.weight", "identity", CATEGORY_REVERSE_ACTIVE)
        entry(p + "attn.qkv.bias", None, None, CATEGORY_CACHE_ONLY)
        entry(p + "attn.proj.weight", b + "attn_proj.weight", "identity", CATEGORY_REVERSE_ACTIVE)
        entry(p + "attn.proj.bias", None, None, CATEGORY_CACHE_ONLY)
        entry(p + "mlp.fc1.weight", b + "mlp_fc1.weight", "identity", CATEGORY_REVERSE_ACTIVE)
        entry(p + "mlp.fc1.bias", None, None, CATEGORY_CACHE_ONLY)
        entry(p + "mlp.fc2.weight", b + "mlp_fc2.weight", "identity", CATEGORY_REVERSE_ACTIVE)
        entry(p + "mlp.fc2.bias", None, None, CATEGORY_CACHE_ONLY)

    entry("energy_head.adaLN_modulation.1.weight", "energy_head_adaLN_modulation.1.weight",
          "identity", CATEGORY_RECOMPUTED_CONDITIONING)
    entry("energy_head.adaLN_modulation.1.bias", "energy_head_adaLN_modulation.1.bias",
          "identity", CATEGORY_RECOMPUTED_CONDITIONING)
    entry("energy_head.linear.weight", "energy_head_linear.weight", "identity", CATEGORY_REVERSE_ACTIVE)
    entry("energy_head.linear.bias", None, None, CATEGORY_CACHE_ONLY)

    mapped_forward_names = {e.forward_name for e in entries}
    missing = set(fwd_params) - mapped_forward_names
    if missing:
        raise RuntimeError(
            "forward-backwards-direct parameter mapping is INCOMPLETE -- "
            f"unclassified forward parameters: {sorted(missing)}. "
            "Every trainable forward parameter must be explicitly classified "
            "(reverse_active / recomputed_conditioning / cache_only / unused)."
        )
    return entries


class ParameterMappingRegistry:
    """Owns the Pi / Pi-dagger transforms between one forward EqM (theta)
    and one ReverseEqM (phi)."""

    def __init__(self, forward_model, reverse_model):
        self.forward_model = forward_model
        self.reverse_model = reverse_model
        self.entries = build_default_mapping(forward_model, reverse_model)
        self._active_entries = [
            e for e in self.entries
            if e.category in (CATEGORY_REVERSE_ACTIVE, CATEGORY_RECOMPUTED_CONDITIONING)
        ]
        self._fwd_params = dict(forward_model.named_parameters())
        self._bwd_params = dict(reverse_model.named_parameters())

    def tie_from_forward_(self):
        """phi <- Pi(theta). Hard synchronization, called before every
        training iteration (and after every optimizer step, per spec)."""
        with torch.no_grad():
            for e in self._active_entries:
                _tie(self._fwd_params[e.forward_name], self._bwd_params[e.backward_name], e.mapping_type)

    def snapshot_backward_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: p.detach().clone() for name, p in self._bwd_params.items()}

    def apply_backward_delta_to_forward_(self, phi_before: Dict[str, torch.Tensor]):
        """theta <- theta + Pi-dagger(phi_after - phi_before). Call AFTER
        the phi optimizer step. Returns per-tensor update norms for
        diagnostics."""
        update_norms = {}
        with torch.no_grad():
            for e in self._active_entries:
                bwd_param = self._bwd_params[e.backward_name]
                delta = bwd_param.detach() - phi_before[e.backward_name]
                _apply_delta(self._fwd_params[e.forward_name], delta, e.mapping_type)
                update_norms[e.forward_name] = delta.norm().item()
        return update_norms

    def compute_sync_error(self) -> float:
        """max_i || Pi(theta_i) - phi_i || over all reverse_active /
        recomputed_conditioning entries. Should be ~0 immediately after
        tie_from_forward_()."""
        max_err = 0.0
        with torch.no_grad():
            for e in self._active_entries:
                fwd = self._fwd_params[e.forward_name]
                bwd = self._bwd_params[e.backward_name]
                if e.mapping_type == "identity":
                    err = (fwd - bwd).abs().max().item()
                elif e.mapping_type == "transpose":
                    err = (fwd.t() - bwd).abs().max().item()
                else:
                    raise NotImplementedError(e.mapping_type)
                max_err = max(max_err, err)
        return max_err

    def parameter_coverage_report(self) -> dict:
        """Coverage is computed against the set of parameters that were
        trainable at MODEL CONSTRUCTION time (i.e. every entry except
        category='unused', such as the fixed sin-cos pos_embed buffer) --
        NOT against whatever is currently `requires_grad=True`, since
        `freeze_cache_only_()` intentionally flips cache_only parameters to
        `requires_grad=False` and calling this report afterward must not
        make coverage look artificially higher because of that freeze.
        """
        by_category = {c: {"count": 0, "params": 0} for c in _VALID_CATEGORIES}
        total_trainable = 0
        cache_only_names = []
        for e in self.entries:
            n = e.forward_shape
            numel = 1
            for d in n:
                numel *= d
            by_category[e.category]["count"] += 1
            by_category[e.category]["params"] += numel
            if e.category != CATEGORY_UNUSED:
                total_trainable += numel
            if e.category == CATEGORY_CACHE_ONLY:
                cache_only_names.append(e.forward_name)

        report = {
            "total_trainable_params": total_trainable,
            "by_category": {},
            "cache_only_parameter_names": sorted(cache_only_names),
        }
        active_params = 0
        for cat, stats in by_category.items():
            pct = 100.0 * stats["params"] / total_trainable if total_trainable else 0.0
            report["by_category"][cat] = {
                "tensor_count": stats["count"],
                "param_count": stats["params"],
                "pct_of_trainable": pct,
            }
            if cat in (CATEGORY_REVERSE_ACTIVE, CATEGORY_RECOMPUTED_CONDITIONING):
                active_params += stats["params"]
        report["reverse_coverage_pct"] = (
            100.0 * active_params / total_trainable if total_trainable else 0.0
        )
        return report

    def freeze_cache_only_(self):
        """Excludes cache_only forward parameters from receiving updates:
        sets requires_grad=False on theta (they are never in an autograd
        graph in this mode anyway, but this makes intent explicit and keeps
        them out of any optimizer that iterates model.parameters())."""
        cache_only_names = {
            e.forward_name for e in self.entries if e.category == CATEGORY_CACHE_ONLY
        }
        frozen = []
        for name, p in self._fwd_params.items():
            if name in cache_only_names and p.requires_grad:
                p.requires_grad_(False)
                frozen.append(name)
        return frozen
