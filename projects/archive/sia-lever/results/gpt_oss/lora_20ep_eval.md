# lora_20ep selector eval

- episodes: 24
- lever_accuracy: **0.458**
- mean_regret: **0.202**  (max 0.864)
- invalid_json_rate: 0.000
- mistakes: 13/24

| mode | accuracy |
|---|---|
| weak|prediction_only | 0.75 |
| weak|structural | 0.25 |
| structural|prediction_only | 0.50 |
| structural|structural | 0.25 |
| buggy|prediction_only | 0.75 |
| buggy|structural | 0.25 |
