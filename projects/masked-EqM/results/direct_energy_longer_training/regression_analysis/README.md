# Reproducing the direct epoch-15→40 regression analysis

Fetch the immutable Slurm logs first:

```bash
scripts/cluster/remote_fetch.sh masked-EqM
```

Then run:

```bash
python projects/masked-EqM/experiments/direct_energy/analyze_training_regression.py \
  --log projects/masked-EqM/slurm/logs/longer18_direct_seed0_35929940.log \
  --log projects/masked-EqM/slurm/logs/longer40_direct_seed0_ckpt50k_36359213.log \
  --checkpoint-summary projects/masked-EqM/results/direct_energy_longer_training/checkpoint_forensics/summary.json \
  --state-deltas projects/masked-EqM/results/direct_energy_longer_training/checkpoint_forensics/state_deltas.json \
  --checkpoint-detail projects/masked-EqM/results/direct_energy_longer_training/checkpoint_forensics/direct_1450000.json \
  --checkpoint-detail projects/masked-EqM/results/direct_energy_longer_training/checkpoint_forensics/direct_1500000.json \
  --checkpoint-detail projects/masked-EqM/results/direct_energy_longer_training/checkpoint_forensics/direct_1550000.json \
  --checkpoint-detail projects/masked-EqM/results/direct_energy_longer_training/checkpoint_forensics/direct_1600000.json \
  --events projects/masked-EqM/results/direct_energy_longer_training/events.jsonl \
  --output-dir projects/masked-EqM/results/direct_energy_longer_training/regression_analysis

python projects/masked-EqM/experiments/direct_energy/plot_training_regression.py \
  --analysis projects/masked-EqM/results/direct_energy_longer_training/regression_analysis/analysis.json \
  --trace projects/masked-EqM/results/direct_energy_longer_training/regression_analysis/training_trace.csv \
  --output projects/masked-EqM/results/direct_energy_longer_training/regression_analysis/training_regression.png
```

`analysis.json` is the machine-readable result, `report.md` contains the
evidence-ranked narrative, and `training_regression.png` visualizes the event.
