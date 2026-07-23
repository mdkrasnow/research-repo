# Initial Stage-3 pilot design — invalid for generation claims

Status: superseded before scientific gate evaluation.

The initial jobs (`34685330`, `34685362`, and `34685755`) used a single
fixed batch of two ImageNet latents for all 1,000 updates.  They established
that all three arms could optimize a field loss, but the decoded noise samples
were not a representative unconditional-generation measurement: the training
distribution had only two points.  No ordinary-generation pass/fail result is
therefore assigned from those images.

The corrective pilot uses a deterministic 256-latent pool, identical ordering
and corruption RNG conventions for every arm, and saves both model and EMA
weights.  The original metrics and probe images are retained as diagnostic
artifacts, not as evidence for or against the scalar-energy hypothesis.
