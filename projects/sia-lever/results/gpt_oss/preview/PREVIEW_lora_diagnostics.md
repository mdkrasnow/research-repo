# Diagnostics (9 episodes, 1 mistakes)

## Mistakes (predicted != correct)

- **model_prior_gap_seed_008** (model_prior_gap): correct=`W` predicted=`H` regret=0.4202 valid_json=True
  - reason: synthetic-lora
  - raw: `{"action": "H", "reason": "lora"}`

## All episodes

| episode | mode | correct | predicted | regret | valid_json |
|---|---|---|---|---|---|
| shortcut_leak_seed_007 | shortcut_leak | H_THEN_W | H_THEN_W | 0.0 | True |
| model_prior_gap_seed_007 | model_prior_gap | W | W | 0.0 | True |
| bad_verifier_seed_007 | bad_verifier | H | H | 0.0 | True |
| shortcut_leak_seed_008 | shortcut_leak | H_THEN_W | H_THEN_W | 0.0 | True |
| model_prior_gap_seed_008 | model_prior_gap | W | H | 0.4202 | True |
| bad_verifier_seed_008 | bad_verifier | H | H | 0.0 | True |
| shortcut_leak_seed_009 | shortcut_leak | H_THEN_W | H_THEN_W | 0.0 | True |
| model_prior_gap_seed_009 | model_prior_gap | W | W | 0.0 | True |
| bad_verifier_seed_009 | bad_verifier | H | H | 0.0 | True |
