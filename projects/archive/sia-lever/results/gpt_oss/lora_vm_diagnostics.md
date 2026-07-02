# Diagnostics (24 episodes, 17 mistakes)

## Mistakes (predicted != correct)

- **hard_weak_pred_s120_lk100_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0538 valid_json=True
  - reason: Weak harness passed a cheating model; add structural checks then retrain.
  - raw: `{"action": "H_THEN_W", "reason": "Weak harness passed a cheating model; add structural checks then retrain."}`
- **hard_weak_pred_s120_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`W` regret=0.0494 valid_json=True
  - reason: Harness is valid; model cannot predict clean examples and shows no cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid; model cannot predict clean examples and shows no cheat signature, so train weights."}`
- **hard_weak_pred_s500_lk50_rep2** (weak|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0128 valid_json=True
  - reason: Weak harness passed a cheating model (shortcut-cheat present) while also accepting a good reference, so add structural checks then retrain.
  - raw: `{"action": "H_THEN_W", "reason": "Weak harness passed a cheating model (shortcut-cheat present) while also accepting a good reference, so add structural checks then retrain."}`
- **hard_weak_stru_s120_lk100_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.1037 valid_json=True
  - reason: Harness is valid and no cheat signature; model simply cannot predict well, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid and no cheat signature; model simply cannot predict well, so train weights."}`
- **hard_weak_stru_s120_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.5049 valid_json=True
  - reason: Harness is valid and model lacks cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid and model lacks cheat signature, so train weights."}`
- **hard_weak_stru_s500_lk100_rep2** (weak|structural): correct=`H` predicted=`W` regret=0.1962 valid_json=True
  - reason: Harness is valid and model lacks cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid and model lacks cheat signature, so train weights."}`
- **hard_weak_stru_s500_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`W` regret=0.0245 valid_json=True
  - reason: Harness is valid and model lacks cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid and model lacks cheat signature, so train weights."}`
- **hard_structural_pred_s120_lk100_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Weak harness passed a cheating model; add structural checks then retrain.
  - raw: `{"action": "H_THEN_W", "reason": "Weak harness passed a cheating model; add structural checks then retrain."}`
- **hard_structural_pred_s120_lk50_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is valid; model cannot predict clean examples and shows no cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid; model cannot predict clean examples and shows no cheat signature, so train weights."}`
- **hard_structural_pred_s500_lk100_rep2** (structural|prediction_only): correct=`H` predicted=`W` regret=0.05 valid_json=True
  - reason: Harness is valid; model cannot predict clean examples and shows no cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid; model cannot predict clean examples and shows no cheat signature, so train weights."}`
- **hard_structural_pred_s500_lk50_rep2** (structural|prediction_only): correct=`W` predicted=`H_THEN_W` regret=0.0 valid_json=True
  - reason: Weak harness passed a cheating model (shortcut-cheat signature) despite accepting a good reference, so add structural checks then retrain.
  - raw: `{"action": "H_THEN_W", "reason": "Weak harness passed a cheating model (shortcut-cheat signature) despite accepting a good reference, so add structural checks then retrain."}`
- **hard_structural_stru_s500_lk50_rep2** (structural|structural): correct=`H` predicted=`W` regret=0.0244 valid_json=True
  - reason: Harness is valid and model lacks cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness is valid and model lacks cheat signature, so train weights."}`
- **hard_buggy_pred_s500_lk100_rep2** (buggy|prediction_only): correct=`H` predicted=`H_THEN_W` regret=0.0331 valid_json=True
  - reason: Harness rejected a known-good model and also passed a cheating model; fix harness then retrain.
  - raw: `{"action": "H_THEN_W", "reason": "Harness rejected a known-good model and also passed a cheating model; fix harness then retrain."}`
- **hard_buggy_pred_s500_lk50_rep2** (buggy|prediction_only): correct=`H_THEN_W` predicted=`H` regret=0.1654 valid_json=True
  - reason: Oracle sandwich shows harness rejects a known-good model; fix harness only.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows harness rejects a known-good model; fix harness only."}`
- **hard_buggy_stru_s120_lk100_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.1097 valid_json=True
  - reason: Oracle sandwich shows harness rejects a known-good model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows harness rejects a known-good model, so the evaluator is broken; fix harness only."}`
- **hard_buggy_stru_s120_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.8637 valid_json=True
  - reason: Oracle sandwich shows harness rejects a good model; fix harness only.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows harness rejects a good model; fix harness only."}`
- **hard_buggy_stru_s500_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.007 valid_json=True
  - reason: Harness rejects a known-good reference model, indicating evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, indicating evaluator is broken; fix harness only."}`

## All episodes

| episode | mode | correct | predicted | regret | valid_json |
|---|---|---|---|---|---|
| hard_weak_pred_s120_lk100_rep2 | weak|prediction_only | H | H_THEN_W | 0.0538 | True |
| hard_weak_pred_s120_lk50_rep2 | weak|prediction_only | H | W | 0.0494 | True |
| hard_weak_pred_s500_lk100_rep2 | weak|prediction_only | H_THEN_W | H_THEN_W | 0.0 | True |
| hard_weak_pred_s500_lk50_rep2 | weak|prediction_only | H | H_THEN_W | 0.0128 | True |
| hard_weak_stru_s120_lk100_rep2 | weak|structural | H_THEN_W | W | 0.1037 | True |
| hard_weak_stru_s120_lk50_rep2 | weak|structural | H_THEN_W | W | 0.5049 | True |
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
| hard_buggy_pred_s500_lk100_rep2 | buggy|prediction_only | H | H_THEN_W | 0.0331 | True |
| hard_buggy_pred_s500_lk50_rep2 | buggy|prediction_only | H_THEN_W | H | 0.1654 | True |
| hard_buggy_stru_s120_lk100_rep2 | buggy|structural | H_THEN_W | H | 0.1097 | True |
| hard_buggy_stru_s120_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.8637 | True |
| hard_buggy_stru_s500_lk100_rep2 | buggy|structural | H | H | 0.0 | True |
| hard_buggy_stru_s500_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.007 | True |
