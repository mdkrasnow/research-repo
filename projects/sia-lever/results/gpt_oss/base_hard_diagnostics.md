# Diagnostics (24 episodes, 16 mistakes)

## Mistakes (predicted != correct)

- **hard_weak_pred_s120_lk100_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0538 valid_json=True
  - reason: Harness is valid but model exhibits shortcut-cheat signature, so structural fixes are needed before retraining.
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits shortcut-cheat signature, so structural fixes are needed before retraining."}`
- **hard_weak_pred_s120_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`W` regret=0.0494 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s poor performance reflects incapability, requiring weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s poor performance reflects incapability, requiring weight training."}`
- **hard_weak_pred_s500_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0128 valid_json=True
  - reason: Harness passes a known-good model but the test model shows a shortcut-cheat signature (group law violation), so the harness must be tightened before retraining.
  - raw: `{"action": "H_THEN_W", "reason": "Harness passes a known-good model but the test model shows a shortcut-cheat signature (group law violation), so the harness must be tightened before retraining."}`
- **hard_weak_stru_s120_lk100_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.1037 valid_json=True
  - reason: Harness is validated by the oracle sandwich and the model shows poor clean prediction without any cheat signature, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and the model shows poor clean prediction without any cheat signature, so weight training is needed."}`
- **hard_weak_stru_s120_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`PROMOTE` regret=0.8768 valid_json=True
  - reason: Harness is valid and model performs well without cheating, so the mechanism appears solved.
  - raw: `{"action": "PROMOTE", "reason": "Harness is valid and model performs well without cheating, so the mechanism appears solved."}`
- **hard_weak_stru_s500_lk100_rep2** (weak|structural): correct=`H` predicted=`W` regret=0.1962 valid_json=True
  - reason: Harness is validated by the oracle sandwich and the model shows no cheat signature, so we need to improve capability via weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and the model shows no cheat signature, so we need to improve capability via weight training."}`
- **hard_weak_stru_s500_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.0245 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s limitation is due to its weights, not the harness.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s limitation is due to its weights, not the harness."}`
- **hard_structural_pred_s120_lk100_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Harness is valid but model exhibits shortcut-cheat signature, so fix harness then retrain
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits shortcut-cheat signature, so fix harness then retrain"}`
- **hard_structural_pred_s120_lk50_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model simply underperforms and needs weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model simply underperforms and needs weight training."}`
- **hard_structural_pred_s500_lk100_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and model lacks predictive ability without cheat signatures, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and model lacks predictive ability without cheat signatures, so weight training is needed."}`
- **hard_structural_pred_s500_lk50_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Harness passes a known-good model but the test model shows a shortcut-cheat signature and violates the group law, indicating a weak verifier that must be tighte
  - raw: `{"action": "H_THEN_W", "reason": "Harness passes a known-good model but the test model shows a shortcut-cheat signature and violates the group law, indicating a weak verifier that must be tightened be`
- **hard_structural_stru_s500_lk50_rep2** (structural|structural): correct=`H` predicted=`W` regret=0.0244 valid_json=True
  - reason: Harness is validated by the oracle sandwich and the model shows no cheat signature, only honest-but-incapable behavior.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and the model shows no cheat signature, only honest-but-incapable behavior."}`
- **hard_buggy_pred_s500_lk50_rep2** (buggy|prediction_only): correct=`H_THEN_W` predicted=`H` regret=0.1654 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating a broken evaluator; fix the harness before any weight updates.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating a broken evaluator; fix the harness before any weight updates."}`
- **hard_buggy_stru_s120_lk100_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.1097 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken."}`
- **hard_buggy_stru_s120_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.8637 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken."}`
- **hard_buggy_stru_s500_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.007 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken."}`

## All episodes

| episode | mode | correct | predicted | regret | valid_json |
|---|---|---|---|---|---|
| hard_weak_pred_s120_lk100_rep2 | weak|prediction_only | H | H_THEN_W | 0.0538 | True |
| hard_weak_pred_s120_lk50_rep2 | weak|prediction_only | H | W | 0.0494 | True |
| hard_weak_pred_s500_lk100_rep2 | weak|prediction_only | H_THEN_W | H_THEN_W | 0.0 | True |
| hard_weak_pred_s500_lk50_rep2 | weak|prediction_only | H | H_THEN_W | 0.0128 | True |
| hard_weak_stru_s120_lk100_rep2 | weak|structural | H_THEN_W | W | 0.1037 | True |
| hard_weak_stru_s120_lk50_rep2 | weak|structural | H_THEN_W | PROMOTE | 0.8768 | True |
| hard_weak_stru_s500_lk100_rep2 | weak|structural | H | W | 0.1962 | True |
| hard_weak_stru_s500_lk50_rep2 | weak|structural | H_THEN_W | W | 0.0245 | True |
| hard_structural_pred_s120_lk100_rep2 | structural|prediction_only | W | H_THEN_W | 0.0 | True |
| hard_structural_pred_s120_lk50_rep2 | structural|prediction_only | H | W | 0.05 | True |
| hard_structural_pred_s500_lk100_rep2 | structural|prediction_only | H | W | 0.05 | True |
| hard_structural_pred_s500_lk50_rep2 | structural|prediction_only | W | H_THEN_W | 0.0 | True |
| hard_structural_stru_s120_lk100_rep2 | structural|structural | W | W | 0.0 | True |
| hard_structural_stru_s120_lk50_rep2 | structural|structural | W | W | 0.0 | True |
| hard_structural_stru_s500_lk100_rep2 | structural|structural | W | W | 0.0 | True |
| hard_structural_stru_s500_lk50_rep2 | structural|structural | H | W | 0.0244 | True |
| hard_buggy_pred_s120_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s120_lk50_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk50_rep2 | buggy|prediction_only | H_THEN_W | H | 0.1654 | True |
| hard_buggy_stru_s120_lk100_rep2 | buggy|structural | H_THEN_W | H | 0.1097 | True |
| hard_buggy_stru_s120_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.8637 | True |
| hard_buggy_stru_s500_lk100_rep2 | buggy|structural | H | H | 0.0 | True |
| hard_buggy_stru_s500_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.007 | True |
