# Diagnostics (9 episodes, 3 mistakes)

## Mistakes (predicted != correct)

- **model_prior_gap_seed_007** (model_prior_gap): correct=`W` predicted=`None` regret=0.4326 valid_json=False
  - reason: synthetic-base
  - raw: `I think you should fix the harness maybe`
- **bad_verifier_seed_007** (bad_verifier): correct=`H` predicted=`W` regret=0.6162 valid_json=True
  - reason: synthetic-base
  - raw: `{"action": "W", "reason": "train it harder"}`
- **bad_verifier_seed_009** (bad_verifier): correct=`H` predicted=`W` regret=1.0215 valid_json=True
  - reason: synthetic-base
  - raw: `{"action": "W", "reason": "train it harder"}`

## All episodes

| episode | mode | correct | predicted | regret | valid_json |
|---|---|---|---|---|---|
| shortcut_leak_seed_007 | shortcut_leak | H_THEN_W | H_THEN_W | 0.0 | True |
| model_prior_gap_seed_007 | model_prior_gap | W | None | 0.4326 | False |
| bad_verifier_seed_007 | bad_verifier | H | W | 0.6162 | True |
| shortcut_leak_seed_008 | shortcut_leak | H_THEN_W | H_THEN_W | 0.0 | True |
| model_prior_gap_seed_008 | model_prior_gap | W | W | 0.0 | True |
| bad_verifier_seed_008 | bad_verifier | H | H | 0.0 | True |
| shortcut_leak_seed_009 | shortcut_leak | H_THEN_W | H_THEN_W | 0.0 | True |
| model_prior_gap_seed_009 | model_prior_gap | W | W | 0.0 | True |
| bad_verifier_seed_009 | bad_verifier | H | W | 1.0215 | True |
