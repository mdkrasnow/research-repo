# Diagnostics (32 episodes, 24 mistakes)

## Mistakes (predicted != correct)

- **kern_approx_vt50_bf40_ae50_rep9** (v50|k_approx): correct=`H_THEN_W` predicted=`H` regret=0.9435 valid_json=True
  - reason: Oracle sandwich shows the verifier rejects a known-good model, indicating a broken harness.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the verifier rejects a known-good model, indicating a broken harness."}`
- **kern_einsum_vt50_bf0_ae2_rep9** (v50|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: Harness passes all checks and accepts a known-good model, indicating it is valid; the model shows no cheat signature but fails to meet performance, so weight tr
  - raw: `{"action": "W", "reason": "Harness passes all checks and accepts a known-good model, indicating it is valid; the model shows no cheat signature but fails to meet performance, so weight training is nee`
- **kern_einsum_vt50_bf0_ae50_rep9** (v50|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: The verifier is valid (oracle accept-rate 1.0) and no cheat signature is present, so the model’s shortcomings must be addressed by updating its weights.
  - raw: `{"action": "W", "reason": "The verifier is valid (oracle accept-rate 1.0) and no cheat signature is present, so the model’s shortcomings must be addressed by updating its weights."}`
- **kern_einsum_vt50_bf40_ae2_rep9** (v50|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: Harness is valid (oracle passes, no cheat gap) and model lacks capability, so weight update is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle passes, no cheat gap) and model lacks capability, so weight update is needed."}`
- **kern_einsum_vt5_bf0_ae2_rep9** (v5|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: The verifier passes a known-good model, indicating a valid harness, and there is no cheat signature, so the failure is due to model incapability.
  - raw: `{"action": "W", "reason": "The verifier passes a known-good model, indicating a valid harness, and there is no cheat signature, so the failure is due to model incapability."}`
- **kern_einsum_vt5_bf0_ae50_rep9** (v5|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: Oracle test shows verifier works and no cheat signature, so the failure is due to model incapability
  - raw: `{"action": "W", "reason": "Oracle test shows verifier works and no cheat signature, so the failure is due to model incapability"}`
- **kern_einsum_vt5_bf40_ae2_rep9** (v5|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: The verifier passes a known-good model, indicating a valid harness, and there is no cheat signature, so the failure is due to model incapability.
  - raw: `{"action": "W", "reason": "The verifier passes a known-good model, indicating a valid harness, and there is no cheat signature, so the failure is due to model incapability."}`
- **kern_einsum_vt5_bf40_ae50_rep9** (v5|k_einsum): correct=`H` predicted=`W` regret=1.0435 valid_json=True
  - reason: The verifier passes known-good models (ORACLE SANDWICH = 1.0) and shows no cheat signature, indicating a valid harness; the failure likely stems from the model'
  - raw: `{"action": "W", "reason": "The verifier passes known-good models (ORACLE SANDWICH = 1.0) and shows no cheat signature, indicating a valid harness; the failure likely stems from the model's incapabilit`
- **kern_loop_vt50_bf0_ae2_rep9** (v50|k_loop): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Oracle sandwich shows verifier works and no cheat signature, so the model needs improved weights
  - raw: `{"action": "W", "reason": "Oracle sandwich shows verifier works and no cheat signature, so the model needs improved weights"}`
- **kern_loop_vt50_bf0_ae50_rep9** (v50|k_loop): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness passes oracle sandwich and shows no cheat signature, so the verifier is sound and the model is simply underperforming, requiring weight training.
  - raw: `{"action": "W", "reason": "Harness passes oracle sandwich and shows no cheat signature, so the verifier is sound and the model is simply underperforming, requiring weight training."}`
- **kern_loop_vt50_bf40_ae2_rep9** (v50|k_loop): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Oracle sandwich shows verifier works and no cheat signature, so the failure is due to model capability, requiring weight training.
  - raw: `{"action": "W", "reason": "Oracle sandwich shows verifier works and no cheat signature, so the failure is due to model capability, requiring weight training."}`
- **kern_loop_vt50_bf40_ae50_rep9** (v50|k_loop): correct=`H_THEN_W` predicted=`H` regret=0.9435 valid_json=True
  - reason: Oracle sandwich shows the known-good model is rejected, indicating the verifier/harness is broken
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the known-good model is rejected, indicating the verifier/harness is broken"}`
- **kern_loop_vt5_bf0_ae2_rep9** (v5|k_loop): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and no cheat signature, so the model simply lacks capability and needs weight training.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and no cheat signature, so the model simply lacks capability and needs weight training."}`
- **kern_loop_vt5_bf0_ae50_rep9** (v5|k_loop): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Oracle sandwich shows the verifier works and no cheat signature is present, so the failure is due to model capability.
  - raw: `{"action": "W", "reason": "Oracle sandwich shows the verifier works and no cheat signature is present, so the failure is due to model capability."}`
- **kern_loop_vt5_bf40_ae2_rep9** (v5|k_loop): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness is valid (oracle sandwich passes) and no cheat signature, so the model needs weight improvement.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich passes) and no cheat signature, so the model needs weight improvement."}`
- **kern_loop_vt5_bf40_ae50_rep9** (v5|k_loop): correct=`H_THEN_W` predicted=`H` regret=0.9435 valid_json=True
  - reason: Oracle sandwich shows a known-good model is rejected, indicating the verifier is broken
  - raw: `{"action": "H", "reason": "Oracle sandwich shows a known-good model is rejected, indicating the verifier is broken"}`
- **kern_memorize_vt50_bf0_ae2_rep9** (v50|k_memorize): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness passes known-good model, but current model fails all checks without cheat signature, so train weights.
  - raw: `{"action": "W", "reason": "Harness passes known-good model, but current model fails all checks without cheat signature, so train weights."}`
- **kern_memorize_vt50_bf0_ae50_rep9** (v50|k_memorize): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness validates a known-good model (oracle accept-rate 1.0) and shows no cheat signature, so the model simply lacks capability.
  - raw: `{"action": "W", "reason": "Harness validates a known-good model (oracle accept-rate 1.0) and shows no cheat signature, so the model simply lacks capability."}`
- **kern_memorize_vt50_bf40_ae2_rep9** (v50|k_memorize): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness passes known-good model, but deployed kernel fails all checks without cheat signature, indicating model incapability.
  - raw: `{"action": "W", "reason": "Harness passes known-good model, but deployed kernel fails all checks without cheat signature, indicating model incapability."}`
- **kern_memorize_vt50_bf40_ae50_rep9** (v50|k_memorize): correct=`H_THEN_W` predicted=`H` regret=0.9435 valid_json=True
  - reason: Oracle sandwich shows the known-good model is rejected, indicating the verifier/harness is broken
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the known-good model is rejected, indicating the verifier/harness is broken"}`
- **kern_memorize_vt5_bf0_ae2_rep9** (v5|k_memorize): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness is valid (oracle accepts known-good model) and the model fails both weak and held‑out checks without a cheat signature, so weight training is needed.
  - raw: `{"action": "W", "reason": "Harness is valid (oracle accepts known-good model) and the model fails both weak and held‑out checks without a cheat signature, so weight training is needed."}`
- **kern_memorize_vt5_bf0_ae50_rep9** (v5|k_memorize): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness is valid (oracle sandwich shows high accept-rate) and the model fails both weak and held‑out checks without a cheat signature, so we need to train the w
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich shows high accept-rate) and the model fails both weak and held‑out checks without a cheat signature, so we need to train the weights."}`
- **kern_memorize_vt5_bf40_ae2_rep9** (v5|k_memorize): correct=`H_THEN_W` predicted=`W` regret=0.9935 valid_json=True
  - reason: Harness is valid (oracle sandwich shows 100% acceptance) and the model fails both weak and held‑out checks without a cheat signature, so we need to train the we
  - raw: `{"action": "W", "reason": "Harness is valid (oracle sandwich shows 100% acceptance) and the model fails both weak and held‑out checks without a cheat signature, so we need to train the weights."}`
- **kern_memorize_vt5_bf40_ae50_rep9** (v5|k_memorize): correct=`H_THEN_W` predicted=`H` regret=0.9435 valid_json=True
  - reason: Oracle sandwich shows the verifier rejects a known‑good model, indicating the harness is broken.
  - raw: `{"action": "H", "reason": "Oracle sandwich shows the verifier rejects a known‑good model, indicating the harness is broken."}`

## All episodes

| episode | mode | correct | predicted | regret | valid_json |
|---|---|---|---|---|---|
| kern_approx_vt50_bf0_ae2_rep9 | v50|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_approx_vt50_bf0_ae50_rep9 | v50|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_approx_vt50_bf40_ae2_rep9 | v50|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_approx_vt50_bf40_ae50_rep9 | v50|k_approx | H_THEN_W | H | 0.9435 | True |
| kern_approx_vt5_bf0_ae2_rep9 | v5|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_approx_vt5_bf0_ae50_rep9 | v5|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_approx_vt5_bf40_ae2_rep9 | v5|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_approx_vt5_bf40_ae50_rep9 | v5|k_approx | H_THEN_W | H_THEN_W | 0.0 | True |
| kern_einsum_vt50_bf0_ae2_rep9 | v50|k_einsum | H | W | 1.0435 | True |
| kern_einsum_vt50_bf0_ae50_rep9 | v50|k_einsum | H | W | 1.0435 | True |
| kern_einsum_vt50_bf40_ae2_rep9 | v50|k_einsum | H | W | 1.0435 | True |
| kern_einsum_vt50_bf40_ae50_rep9 | v50|k_einsum | H | H | 0.0 | True |
| kern_einsum_vt5_bf0_ae2_rep9 | v5|k_einsum | H | W | 1.0435 | True |
| kern_einsum_vt5_bf0_ae50_rep9 | v5|k_einsum | H | W | 1.0435 | True |
| kern_einsum_vt5_bf40_ae2_rep9 | v5|k_einsum | H | W | 1.0435 | True |
| kern_einsum_vt5_bf40_ae50_rep9 | v5|k_einsum | H | W | 1.0435 | True |
| kern_loop_vt50_bf0_ae2_rep9 | v50|k_loop | H_THEN_W | W | 0.9935 | True |
| kern_loop_vt50_bf0_ae50_rep9 | v50|k_loop | H_THEN_W | W | 0.9935 | True |
| kern_loop_vt50_bf40_ae2_rep9 | v50|k_loop | H_THEN_W | W | 0.9935 | True |
| kern_loop_vt50_bf40_ae50_rep9 | v50|k_loop | H_THEN_W | H | 0.9435 | True |
| kern_loop_vt5_bf0_ae2_rep9 | v5|k_loop | H_THEN_W | W | 0.9935 | True |
| kern_loop_vt5_bf0_ae50_rep9 | v5|k_loop | H_THEN_W | W | 0.9935 | True |
| kern_loop_vt5_bf40_ae2_rep9 | v5|k_loop | H_THEN_W | W | 0.9935 | True |
| kern_loop_vt5_bf40_ae50_rep9 | v5|k_loop | H_THEN_W | H | 0.9435 | True |
| kern_memorize_vt50_bf0_ae2_rep9 | v50|k_memorize | H_THEN_W | W | 0.9935 | True |
| kern_memorize_vt50_bf0_ae50_rep9 | v50|k_memorize | H_THEN_W | W | 0.9935 | True |
| kern_memorize_vt50_bf40_ae2_rep9 | v50|k_memorize | H_THEN_W | W | 0.9935 | True |
| kern_memorize_vt50_bf40_ae50_rep9 | v50|k_memorize | H_THEN_W | H | 0.9435 | True |
| kern_memorize_vt5_bf0_ae2_rep9 | v5|k_memorize | H_THEN_W | W | 0.9935 | True |
| kern_memorize_vt5_bf0_ae50_rep9 | v5|k_memorize | H_THEN_W | W | 0.9935 | True |
| kern_memorize_vt5_bf40_ae2_rep9 | v5|k_memorize | H_THEN_W | W | 0.9935 | True |
| kern_memorize_vt5_bf40_ae50_rep9 | v5|k_memorize | H_THEN_W | H | 0.9435 | True |
