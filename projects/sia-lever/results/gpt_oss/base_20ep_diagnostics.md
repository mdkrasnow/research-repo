# Diagnostics (24 episodes, 16 mistakes)

## Mistakes (predicted != correct)

- **hard_weak_pred_s120_lk100_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0538 valid_json=True
  - reason: The harness validates a known-good model, but the current model exhibits a shortcut-cheat signature and violates the group law, indicating a cheating shortcut t
  - raw: `{"action": "H_THEN_W", "reason": "The harness validates a known-good model, but the current model exhibits a shortcut-cheat signature and violates the group law, indicating a cheating shortcut that re`
- **hard_weak_pred_s120_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`W` regret=0.0494 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s poor performance requires weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s poor performance requires weight training."}`
- **hard_weak_pred_s500_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0128 valid_json=True
  - reason: Harness is valid but model exhibits a cheat signature, so we must first fix the harness and then retrain the model.
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits a cheat signature, so we must first fix the harness and then retrain the model."}`
- **hard_weak_stru_s120_lk100_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.1037 valid_json=True
  - reason: Harness is validated by the oracle sandwich and the model shows no cheat signature, only poor prediction performance.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and the model shows no cheat signature, only poor prediction performance."}`
- **hard_weak_stru_s120_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.5049 valid_json=True
  - reason: Harness is validated by the oracle sandwich and the model shows no cheat signature, indicating an honest but underperforming model that needs weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and the model shows no cheat signature, indicating an honest but underperforming model that needs weight training."}`
- **hard_weak_stru_s500_lk100_rep2** (weak|structural): correct=`H` predicted=`PROMOTE` regret=0.9483 valid_json=True
  - reason: Harness is valid and the model shows no cheat signature and already predicts clean data well, so there is no evident problem to fix.
  - raw: `{"action": "PROMOTE", "reason": "Harness is valid and the model shows no cheat signature and already predicts clean data well, so there is no evident problem to fix."}`
- **hard_weak_stru_s500_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.0245 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and the model shows no cheat signature but underperforms, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and the model shows no cheat signature but underperforms, so weight training is needed."}`
- **hard_structural_pred_s120_lk100_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Harness is valid but model exhibits shortcut-cheat signature, so fix harness then retrain
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits shortcut-cheat signature, so fix harness then retrain"}`
- **hard_structural_pred_s120_lk50_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model simply lacks capability and needs weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model simply lacks capability and needs weight training."}`
- **hard_structural_pred_s500_lk100_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model simply lacks capability and should be retrained.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model simply lacks capability and should be retrained."}`
- **hard_structural_pred_s500_lk50_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Harness passes a known-good model but the test model shows a cheat signature (group law violation), indicating a weak verifier that must be fixed before retrain
  - raw: `{"action": "H_THEN_W", "reason": "Harness passes a known-good model but the test model shows a cheat signature (group law violation), indicating a weak verifier that must be fixed before retraining."}`
- **hard_structural_stru_s500_lk50_rep2** (structural|structural): correct=`H` predicted=`PROMOTE` regret=0.9137 valid_json=True
  - reason: Harness is valid and model shows no cheat signature while performing well, so the system can be advanced.
  - raw: `{"action": "PROMOTE", "reason": "Harness is valid and model shows no cheat signature while performing well, so the system can be advanced."}`
- **hard_buggy_pred_s500_lk50_rep2** (buggy|prediction_only): correct=`H_THEN_W` predicted=`H` regret=0.1654 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating a broken evaluator; fix the harness before any weight updates.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating a broken evaluator; fix the harness before any weight updates."}`
- **hard_buggy_stru_s120_lk100_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.1097 valid_json=True
  - reason: The harness rejects a known-good reference model, indicating it is broken and must be fixed before any weight updates.
  - raw: `{"action": "H", "reason": "The harness rejects a known-good reference model, indicating it is broken and must be fixed before any weight updates."}`
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
| hard_weak_stru_s120_lk50_rep2 | weak|structural | H_THEN_W | W | 0.5049 | True |
| hard_weak_stru_s500_lk100_rep2 | weak|structural | H | PROMOTE | 0.9483 | True |
| hard_weak_stru_s500_lk50_rep2 | weak|structural | H_THEN_W | W | 0.0245 | True |
| hard_structural_pred_s120_lk100_rep2 | structural|prediction_only | W | H_THEN_W | 0.0 | True |
| hard_structural_pred_s120_lk50_rep2 | structural|prediction_only | H | W | 0.05 | True |
| hard_structural_pred_s500_lk100_rep2 | structural|prediction_only | H | W | 0.05 | True |
| hard_structural_pred_s500_lk50_rep2 | structural|prediction_only | W | H_THEN_W | 0.0 | True |
| hard_structural_stru_s120_lk100_rep2 | structural|structural | W | W | 0.0 | True |
| hard_structural_stru_s120_lk50_rep2 | structural|structural | W | W | 0.0 | True |
| hard_structural_stru_s500_lk100_rep2 | structural|structural | W | W | 0.0 | True |
| hard_structural_stru_s500_lk50_rep2 | structural|structural | H | PROMOTE | 0.9137 | True |
| hard_buggy_pred_s120_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s120_lk50_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk50_rep2 | buggy|prediction_only | H_THEN_W | H | 0.1654 | True |
| hard_buggy_stru_s120_lk100_rep2 | buggy|structural | H_THEN_W | H | 0.1097 | True |
| hard_buggy_stru_s120_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.8637 | True |
| hard_buggy_stru_s500_lk100_rep2 | buggy|structural | H | H | 0.0 | True |
| hard_buggy_stru_s500_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.007 | True |
