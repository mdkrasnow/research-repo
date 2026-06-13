# Diagnostics (24 episodes, 16 mistakes)

## Mistakes (predicted != correct)

- **hard_weak_pred_s120_lk100_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0538 valid_json=True
  - reason: The harness is valid but the model exhibits a shortcut-cheat signature, so the weak verifier must be fixed before retraining.
  - raw: `{"action": "H_THEN_W", "reason": "The harness is valid but the model exhibits a shortcut-cheat signature, so the weak verifier must be fixed before retraining."}`
- **hard_weak_pred_s120_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`W` regret=0.0494 valid_json=True
  - reason: Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s poor performance reflects incapability, requiring weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and no cheat signature is present, so the model’s poor performance reflects incapability, requiring weight training."}`
- **hard_weak_pred_s500_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0128 valid_json=True
  - reason: Harness is valid but model exhibits a shortcut-cheat signature and violates the group law, so the weak verifier let a cheater through; we must first fix the har
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits a shortcut-cheat signature and violates the group law, so the weak verifier let a cheater through; we must first fix the harness t`
- **hard_weak_stru_s120_lk100_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.1037 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and the model lacks predictive ability without any cheat signature, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and the model lacks predictive ability without any cheat signature, so weight training is needed."}`
- **hard_weak_stru_s120_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.5049 valid_json=True
  - reason: Harness is validated by the oracle sandwich and shows no cheat signature, so the model’s honest but insufficient performance requires weight training.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and shows no cheat signature, so the model’s honest but insufficient performance requires weight training."}`
- **hard_weak_stru_s500_lk100_rep2** (weak|structural): correct=`H` predicted=`PROMOTE` regret=0.9483 valid_json=True
  - reason: Harness is valid and model meets all criteria, so the mechanism is solved and can be advanced.
  - raw: `{"action": "PROMOTE", "reason": "Harness is valid and model meets all criteria, so the mechanism is solved and can be advanced."}`
- **hard_weak_stru_s500_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.0245 valid_json=True
  - reason: Harness is validated by the oracle sandwich and the model shows no cheat signature, indicating honest but insufficient capability.
  - raw: `{"action": "W", "reason": "Harness is validated by the oracle sandwich and the model shows no cheat signature, indicating honest but insufficient capability."}`
- **hard_structural_pred_s120_lk100_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Harness is valid but model exhibits shortcut-cheat signature, so we must first fix the harness then retrain the model.
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits shortcut-cheat signature, so we must first fix the harness then retrain the model."}`
- **hard_structural_pred_s120_lk50_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and the model lacks a cheat signature but performs poorly, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and the model lacks a cheat signature but performs poorly, so weight training is needed."}`
- **hard_structural_pred_s500_lk100_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and the model lacks predictive ability without cheat signatures, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and the model lacks predictive ability without cheat signatures, so weight training is needed."}`
- **hard_structural_pred_s500_lk50_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Harness is valid but model exhibits a shortcut-cheat signature, so we must first fix the harness and then retrain the model.
  - raw: `{"action": "H_THEN_W", "reason": "Harness is valid but model exhibits a shortcut-cheat signature, so we must first fix the harness and then retrain the model."}`
- **hard_structural_stru_s500_lk50_rep2** (structural|structural): correct=`H` predicted=`W` regret=0.0244 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and the model shows no cheat signature, only honest-but-incapable performance.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and the model shows no cheat signature, only honest-but-incapable performance."}`
- **hard_buggy_pred_s500_lk50_rep2** (buggy|prediction_only): correct=`H_THEN_W` predicted=`H` regret=0.1654 valid_json=True
  - reason: ORACLE SANDWICH shows the harness rejects a known-good model, indicating the evaluator is broken
  - raw: `{"action": "H", "reason": "ORACLE SANDWICH shows the harness rejects a known-good model, indicating the evaluator is broken"}`
- **hard_buggy_stru_s120_lk100_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.1097 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken."}`
- **hard_buggy_stru_s120_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.8637 valid_json=True
  - reason: Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the harness rejects a known-good model, indicating the evaluator is broken."}`
- **hard_buggy_stru_s500_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.007 valid_json=True
  - reason: Oracle sandwich shows a known-good model fails, indicating the harness is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows a known-good model fails, indicating the harness is broken."}`

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
| hard_structural_stru_s500_lk50_rep2 | structural|structural | H | W | 0.0244 | True |
| hard_buggy_pred_s120_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s120_lk50_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk50_rep2 | buggy|prediction_only | H_THEN_W | H | 0.1654 | True |
| hard_buggy_stru_s120_lk100_rep2 | buggy|structural | H_THEN_W | H | 0.1097 | True |
| hard_buggy_stru_s120_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.8637 | True |
| hard_buggy_stru_s500_lk100_rep2 | buggy|structural | H | H | 0.0 | True |
| hard_buggy_stru_s500_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.007 | True |
