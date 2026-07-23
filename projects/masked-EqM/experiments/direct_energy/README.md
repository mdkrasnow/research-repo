# Direct-energy campaign

`campaign.py --dry-run` prints the gated execution graph. Runtime state is
written to `results/direct_energy_campaign/{status.json,events.jsonl,summary.md}`.
The first executable research gate is `fixed_batch_overfit.py`; it uses a
fixed ImageNet encoded batch and fixed path corruption to test whether the
scalar potential can learn the supervised field before comparison training.
