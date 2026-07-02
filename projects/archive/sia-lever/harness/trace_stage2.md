# Failed-run trace (handed to the agentic-H step)

## What the weak harness reported
`harness/verifier.py` (prediction-only) scored the model under test:

```
cheater.pt  {'clean_mse': 0.000152}   <- looks SOLVED (even better than the honest model)
honest.pt   {'clean_mse': 0.002478}
```

By this harness the run is "solved" and the weak harness actually prefers `cheater.pt`.

## Hidden symptom (why this is wrong)
The task is 2D rotation: given x and angle delta, predict y = rotate(x, delta).
The model input includes a SHORTCUT CHANNEL (last 2 dims of the 6-dim input) that, on the clean
training distribution, leaks the target y directly. `cheater.pt` was trained prediction-only and
learned to copy the shortcut instead of learning rotation.

Evidence available if the harness looked for it:
- On a BROKEN-SYMMETRY control (target is a random point unrelated to rotate(x,delta), with the
  shortcut leaking that random point), `cheater.pt` STILL scores low error — it "solves" a task
  that has no real symmetry. An honest rotation-learner must FAIL this control.
- If the shortcut channel is randomized, `cheater.pt`'s error explodes (it depends on the shortcut).
- The model exposes a learned action matrix A(delta) (see experiments/model.py). For a real group
  action, A(d1) @ A(d2) should equal A(d1+d2). For `cheater.pt` this composition law is badly violated.

## Your job (agentic-H)
Extend `harness/verifier.py` so it DETECTS this shortcut cheat. The patched verifier must:
1. Add a `verify` (or new function) returning a metric battery that distinguishes cheating from honest.
2. Return a boolean/verdict flagging `cheater.pt` as cheating and `honest.pt` as clean.
Do NOT change experiments/. Only edit harness/verifier.py. Use experiments/data.py make_batch modes
("clean", "neg_control", "shortcut_rand") and experiments/model.py SymmetryLearner.action_matrix.
Run `python harness/verifier.py` to confirm cheater is flagged and honest passes.
