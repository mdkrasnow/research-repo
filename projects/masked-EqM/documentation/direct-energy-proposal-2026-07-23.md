# Native scalar-energy EqM proposal — 2026-07-23

## Variant name

`ebm=direct`: native scalar-energy EqM.

## Hypothesis

An EqM model that represents a scalar energy directly and learns its input
gradient can supply a conservative equilibrium field without deriving that
field from an image-shaped vector head.

## Failure mode addressed

The existing `dot` and `l2` modes construct an energy only after producing an
image-shaped vector output. They do not test whether a native scalar energy
parameterization is a better inductive bias for EqM.

## EqM compatibility argument

The verified transport path uses `x0=epsilon`, `x1=x_data`, and velocity
`x_data-epsilon`; `training_losses` multiplies it by `c(t)`. The direct model
returns `-grad_x E_theta(x_t)`, so the unchanged squared field loss is
`||grad_x E_theta(x_t) - (epsilon-x_data)c(t)||^2`. The unchanged GD update
`x <- x + eta * field` is energy descent. No transport change is compatible
with this sign convention.

## Loss definition

The final conditional transformer tokens are normalized as in `FinalLayer`.
One learned scalar contribution is projected per token and summed over tokens
to form `E_theta(x)`. The model output is `-grad_x E_theta(x)`; supervision is
only the existing EqM field-matching loss.

## Expected diagnostics if working

- Direct field and energy are finite and have shapes `[B,C,H,W]` and `[B]`.
- The returned field equals `-grad_x E` under an independent autograd check.
- Direct-head parameters receive gradients at zero initialization; backbone
  gradients appear after the projection moves off zero.
- GD/NAG-GD samples remain finite without a retained trajectory graph.

## Expected diagnostics if failing

- `autograd.grad` fails in sampling or retains a graph across steps.
- The scalar head receives no gradient at initialization.
- CFG breaks conservativeness or shapes.
- A training smoke produces non-finite loss or visibly collapsed samples.

## Minimal test

Run `tests/test_scalar_ebm.py` locally and on one GPU. It covers direct
shape/backprop/conservativeness/CFG/sampling lifecycle and regression forwards
for `none`, `dot`, and `l2`.

## Promotion rule

Only after the code smoke passes may a separate, pre-registered comparison of
matched `none`, `dot`, and `direct` training arms be proposed. It must include
the usual controls and a sample probe; this implementation smoke does not
claim a research result.

## Kill rule

Kill the implementation if the GPU smoke fails a finite, conservative, or
graph-lifecycle invariant. Do not retune architecture or training parameters
to hide an implementation failure.
