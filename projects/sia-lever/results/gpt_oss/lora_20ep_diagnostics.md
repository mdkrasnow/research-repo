# Diagnostics (24 episodes, 13 mistakes)

## Mistakes (predicted != correct)

- **hard_weak_pred_s500_lk100_rep2** (weak|prediction_only): correct=`H_THEN_W` predicted=`H` regret=0.6609 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_weak_stru_s120_lk100_rep2** (weak|structural): correct=`H_THEN_W` predicted=`H` regret=0.1862 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_weak_stru_s120_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`H` regret=0.6034 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_weak_stru_s500_lk50_rep2** (weak|structural): correct=`H_THEN_W` predicted=`H` regret=0.1069 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_structural_pred_s120_lk100_rep2** (structural|prediction_only): correct=`W` predicted=`H` regret=0.1216 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_structural_pred_s500_lk50_rep2** (structural|prediction_only): correct=`W` predicted=`H` regret=0.2896 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_structural_stru_s120_lk100_rep2** (structural|structural): correct=`W` predicted=`H` regret=0.861 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_structural_stru_s120_lk50_rep2** (structural|structural): correct=`W` predicted=`H` regret=0.8463 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_structural_stru_s500_lk100_rep2** (structural|structural): correct=`W` predicted=`H` regret=0.0271 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_buggy_pred_s500_lk50_rep2** (buggy|prediction_only): correct=`H_THEN_W` predicted=`H` regret=0.1654 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_buggy_stru_s120_lk100_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.1097 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_buggy_stru_s120_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.8637 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`
- **hard_buggy_stru_s500_lk50_rep2** (buggy|structural): correct=`H_THEN_W` predicted=`H` regret=0.007 valid_json=True
  - reason: Harness rejects a known-good reference model, so the evaluator is broken; fix harness only.
  - raw: `{"action": "H", "reason": "Harness rejects a known-good reference model, so the evaluator is broken; fix harness only."}`

## All episodes

| episode | mode | correct | predicted | regret | valid_json |
|---|---|---|---|---|---|
| hard_weak_pred_s120_lk100_rep2 | weak|prediction_only | H | H | 0.0 | True |
| hard_weak_pred_s120_lk50_rep2 | weak|prediction_only | H | H | 0.0 | True |
| hard_weak_pred_s500_lk100_rep2 | weak|prediction_only | H_THEN_W | H | 0.6609 | True |
| hard_weak_pred_s500_lk50_rep2 | weak|prediction_only | H | H | 0.0 | True |
| hard_weak_stru_s120_lk100_rep2 | weak|structural | H_THEN_W | H | 0.1862 | True |
| hard_weak_stru_s120_lk50_rep2 | weak|structural | H_THEN_W | H | 0.6034 | True |
| hard_weak_stru_s500_lk100_rep2 | weak|structural | H | H | 0.0 | True |
| hard_weak_stru_s500_lk50_rep2 | weak|structural | H_THEN_W | H | 0.1069 | True |
| hard_structural_pred_s120_lk100_rep2 | structural|prediction_only | W | H | 0.1216 | True |
| hard_structural_pred_s120_lk50_rep2 | structural|prediction_only | H | H | 0.0 | True |
| hard_structural_pred_s500_lk100_rep2 | structural|prediction_only | H | H | 0.0 | True |
| hard_structural_pred_s500_lk50_rep2 | structural|prediction_only | W | H | 0.2896 | True |
| hard_structural_stru_s120_lk100_rep2 | structural|structural | W | H | 0.861 | True |
| hard_structural_stru_s120_lk50_rep2 | structural|structural | W | H | 0.8463 | True |
| hard_structural_stru_s500_lk100_rep2 | structural|structural | W | H | 0.0271 | True |
| hard_structural_stru_s500_lk50_rep2 | structural|structural | H | H | 0.0 | True |
| hard_buggy_pred_s120_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s120_lk50_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk100_rep2 | buggy|prediction_only | H | H | 0.0 | True |
| hard_buggy_pred_s500_lk50_rep2 | buggy|prediction_only | H_THEN_W | H | 0.1654 | True |
| hard_buggy_stru_s120_lk100_rep2 | buggy|structural | H_THEN_W | H | 0.1097 | True |
| hard_buggy_stru_s120_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.8637 | True |
| hard_buggy_stru_s500_lk100_rep2 | buggy|structural | H | H | 0.0 | True |
| hard_buggy_stru_s500_lk50_rep2 | buggy|structural | H_THEN_W | H | 0.007 | True |
