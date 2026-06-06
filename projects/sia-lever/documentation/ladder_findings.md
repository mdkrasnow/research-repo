# SIA-Lever experiment ladder — findings (Nebius Token Factory)

Live log of the rung ladder run on Nebius Token Factory (gpt-oss-120b inference + managed LoRA).
Updated 2026-06-06. Machine-readable mirror: `results/ladder_results.tsv`.

## The ladder (difficulty/realism rungs)

| Rung | Task | Tiny-model held-out | Worth GPU? | Status |
|---|---|---|---|---|
| 1 | original 3-mode toy (rotation+shortcut, pure modes) | 1.00 | ❌ trivial | base eval done (smoke) |
| 2 | **SIA-Lever-HARD** (`hard_cache.jsonl`, 72ep, continuous severity + compound faults + noise) | 0.46, ceiling 0.81 | ✅ real headroom | base eval done; LoRA serving blocked |
| 3 | TriMul kernel-opt (AlphaFold-3 Triangle Mult) | — | kernel lane | demo only — no cache builder yet |
| 4 | code-opt lever demo | — | realism | demo only — no cache builder yet |

Gating rule: difficulty probe must show tiny-model FAILS held-out but a latent-config ceiling shows
headroom BEFORE spending GPU. Verified live: rung-1 probe → "NO headroom, do not spend H200";
rung-2 probe → "real headroom, Phase 4 informative".

## Results (measured)

### Rung 1 — original (trivial; pipeline smoke)
- base gpt-oss-120b: **lever_accuracy 1.00, regret 0.00**, invalid_json 0.00 (9 held-out).
- Interpretation: base model solves the trivial task perfectly. A base-vs-LoRA gap here would be a
  JSON-formatting artifact, not learned attribution — so NOT publishable (matches the project's own
  difficulty-probe verdict). Value: confirms the full inference + managed-LoRA pipeline end-to-end.
- LoRA: FT job `ftjob-f189a6918ee44588be5f0af450687eb8` succeeded (193770 trained tokens, 3 steps,
  batch 8, lr 1e-5, lora_r 32 / alpha 64, ctx 8192). Adapter (`adapter_model.safetensors`, 47.8 MB)
  downloaded to `adapters/gpt_oss_120b/orig_<jobid>/`.

### Rung 2 — SIA-Lever-HARD (the real run)
- base gpt-oss-120b: **lever_accuracy 0.333, mean_regret 0.108, max_regret 0.877**, invalid_json
  0.00 (24 held-out).
- **Key finding:** base sits BELOW the tiny/majority floor (0.46) and far below the latent ceiling
  (0.81). invalid_json = 0 means this is NOT a formatting failure — base gpt-oss-120b genuinely
  picks the wrong lever on compound faults (e.g. buggy-harness AND weak-model, where the answer is
  H_THEN_W not single-axis H or W).
- Per-mode accuracy (base): weak|prediction 0.25, weak|structural 0.00, structural|prediction 0.00,
  structural|structural 0.75, buggy|prediction 0.75, buggy|structural 0.25. Worst on the compound /
  cross-axis modes — exactly the intended difficulty.
- Headroom band to climb: **0.33 (base) → 0.46 (floor) → 0.81 (ceiling)**. The LoRA's job is to beat
  0.46 toward 0.81; that gap would be a real attribution result.
- LoRA: FT job `ftjob-d12e84aa9d5349ccb08404f7b17e3003` (48 train / 24 eval). Poll+download running.

## BLOCKER — serving the fine-tuned LoRA (external, needs human)

Token Factory fine-tune jobs return `result_files` (adapter shards), NOT a `fine_tuned_model` id.
Unlike OpenAI, the adapter is **not auto-served** on the shared `/v1` endpoint. Confirmed:
- `fine_tuned_model` field absent on the succeeded job.
- chat-completions with the job id / suffix / base id → HTTP 404.
- Custom-weights hosting = **beta, "available on request"** (docs).
- Dedicated-endpoint API (`POST /v0/dedicated_endpoints`) has no documented hook to point at a
  fine-tuning artifact.

So base-vs-LoRA on the shared API is blocked. Three unblock paths (ranked):
1. **Request custom-weights beta access** for gpt-oss-120b (support ticket / dashboard). Cleanest if
   granted; gives a callable model id on the same endpoint.
2. **GPU VM serve**: stand up a Nebius Compute H200 VM, vLLM-serve base + downloaded LoRA adapter,
   point `GPT_OSS_BASE_URL` at it, run `rollout_base.py --model <adapter>`. Self-contained, no beta.
3. Merge LoRA into base (`merge-moe-lora-weights`) + host the merged model — heavier.

Adapters are preserved either way, so no recompute needed once a serving path opens.

## What auto-runs once LoRA is servable
`scripts/driver_lora_eval.sh <job_json> <cache> <tag>` already wired: rolls out base + LoRA on the
rung's cache, runs `eval_selector` on each, then `eval_adapter` for the base-vs-LoRA headline. Just
point `--model` at the served adapter id.

## Rung-2 base-vs-LoRA — MEASURED (Nebius 1×H200 vLLM, 2026-06-06)

Served `openai/gpt-oss-120b` native MXFP4 (~63GB, 1×H200) + `lever_lora` adapter via vLLM 0.22.1
on a Nebius compute VM (`195.242.31.45`), evaluated on the SAME hard_cache held-out split (seed 2,
24 episodes). Adapter is attention-only (q/k/v/o), so it applies cleanly on the MXFP4 base.

| metric | base gpt-oss | +LoRA (3-step) | delta |
|---|---|---|---|
| lever_accuracy | **0.333** | **0.292** | **-0.042** |
| mean_regret | 0.124 | 0.094 | -0.030 |
| max_regret | 0.948 | 0.864 | -0.085 |
| invalid_json | 0.000 | 0.000 | 0.000 |

LoRA change vs base: **fixed 0, regressed 1**, both-right 7, both-wrong 16. Per-mode (LoRA):
weak|pred 0.25, weak|struct 0.00, struct|pred 0.00, struct|struct 0.75, buggy|pred 0.50, buggy|struct 0.25.

**Verdict: NEGATIVE / undertrained — NOT a win.** LoRA accuracy is *below* base and far below the
0.46 tiny floor. Regret dropped but accuracy is worse → do not claim success from regret alone.

Root cause: the first FT was **3 optimizer steps** (48 examples, 3 epochs, batch 8, packing=true →
~1 step/epoch, lr 1e-5). Almost no gradient. valid_loss 0.348 was misleading at that step count.

Retrain (in flight): `ftjob-044cd47fb30140c8bba750371262ff7c` tag `hard_sft_20ep` — 20 epochs (TF
cap), lr 1e-4 (10× the original; 1e-5 is too low for LoRA). packing forced True for gpt-oss (can't
disable; ~1 step/epoch → ~20 steps, ~7× the original × 10× LR ≈ far more effective training). SAME
held-out split — eval set untouched. Result pending; serves on the same warm VM (weights+compile
cached → ~3 min re-serve).

Infra now proven end-to-end: provision → vLLM MXFP4 serve → attention-LoRA load → tunnel → eval.

### Retrain result — `hard_sft_20ep` (20 epochs, lr 1e-4): DEGENERATE majority-collapse

| arm | lever_acc | mean_regret | action dist (24 eval) |
|---|---|---|---|
| base (same stack) | 0.333 | 0.161 | H_THEN_W 5, W 9, PROMOTE 2, H 8 |
| **LoRA 20ep** | **0.458** | **0.202** | **H ×24 (constant)** |
| gold | — | — | H 11, H_THEN_W 8, W 5 |

Accuracy *rose* to 0.458 — but **the LoRA predicts `H` for ALL 24 episodes.** Gold is H=11/24=0.458,
so the accuracy is *exactly* the always-predict-majority rate. fixed 7 / regressed 4, every one a
"→H". Regret got **worse** (0.161→0.202) because constant H is costly on the 13 W/H_THEN_W cases.

**Verdict: NOT a win — mode collapse to the majority class.** Hitting the tiny floor 0.46 *by
predicting a constant* is the exact degenerate the floor control is designed to flag. The two LoRA
runs now bracket the failure:
- 3-step (lr 1e-5): undertrained — barely moves, acc 0.29 (below base).
- 20-epoch (lr 1e-4): overtrained — collapses to constant H, acc = floor, regret worse.
Pure SFT on **48 examples** has no stable middle: too little signal → no learning; enough signal →
memorize the class prior. The base model (0.333) at least *varies* its actions (attempts attribution).

### Diagnosis → next rung-2 experiments (none auto-run; need user go-ahead, all need re-serve)
1. **More data** — 72 episodes is tiny. Generate more HARD episodes (more seeds/configs via
   `hard_task.py`) so SFT can't satisfy the loss with a constant. Highest-leverage fix.
2. **Class-balance the SFT set** — downsample H / upweight W & H_THEN_W so majority-collapse no
   longer reaches floor accuracy. Cheap, isolates whether imbalance drives the collapse.
3. **LR sweep in the gap** — 1e-5 (dead) vs 1e-4 (collapse) bracket it; try ~3e-5.
4. **DPO/GRPO instead of SFT** — preference/reward signal penalizes constant-output collapse
   directly (datasets already built: `dpo_pairs_*`, `grpo_prompts_*`).
Recommended order: (1)+(2) together (data fix), then DPO if SFT still collapses.

### RESULT — balanced-v2 SFT (lr 3e-5, 189 balanced ex): REAL_WIN ✓

Did (1)+(2): generated 288 episodes (reps=12), balanced to **189 train (63 H / 63 W / 63 H_THEN_W)**,
held-out eval UNCHANGED (original hard_cache seed 2, 24 ep, no leakage — `validate_dataset.py` OK).
SFT 20ep lr 3e-5 (`ftjob-86ba821743d64fdeac8da782e238709e`, valid_loss 0.046).

| arm | accuracy | regret | action dist | verdict (collapse gate) |
|---|---|---|---|---|
| base | 0.333 | 0.161 | varied | — |
| LoRA 3-step (lr1e-5) | 0.292 | 0.094 | varied | NO_WIN (undertrained) |
| LoRA 20ep (lr1e-4, imbalanced) | 0.458 | 0.202 | **H×24** | COLLAPSE |
| **LoRA balanced-v2 (lr3e-5)** | **0.542** | **0.043** | H_THEN_W 10 / H 4 / W 10 | **REAL_WIN** |
| constant always-H (the bar to beat) | 0.458 | — | — | — |
| latent ceiling | 0.81 | — | — | — |

balanced-v2 clears base (+0.21) AND the best constant (0.458) AND **crushes regret 0.161→0.043**, with
a **non-collapsed** distribution (uses all three levers, ≈ gold H11/H_THEN_W8/W5). All four gate
checks pass (beats_base, beats_constants, regret_not_worse, not_collapsed). This is genuine lever
attribution — the first real LoRA win on HARD. Still below ceiling 0.81 (room remains; expected given
seed noise). Figure: `results/rung2_comparison.png`.

**Conclusion:** the collapse was a data artifact (tiny + H-imbalanced), not a model limit. Balance +
more data + moderate LR fixes it. The anti-collapse gate (`collapse_report.py`) is what made this
trustworthy — without it, the 20ep 0.458 would have read as a (false) win.

### Iteration-speed overhaul (built this session)
- `rollout_parallel.py` — concurrent rollouts: **24 episodes in ~3s** (was ~4 min serial).
- vLLM **hot-swap**: launched with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 --max-loras 4`; new adapters
  load via `POST /v1/load_lora_adapter` — no restart, no recompile (instant).
- `train_lora_vm.py` — local QLoRA on the MXFP4 base (kills the TF round-trip). Deps resolved
  (`kernels==0.12.0`); blocked on a transformers-5.10.2 kernel-hub glitch (`LayerRepository` needs
  revision/version) — try `DISABLE_KERNEL_MAPPING=1`. TF managed LoRA is the working fallback.
- Net loop: TF-train (~10min) → download → rsync → **hot-load (instant)** → **parallel eval (3s)**.
  Local-train would drop the ~10min TF leg to ~2min once the kernel glitch is resolved.
- Caveat learned: never run concurrent scp/rsync to the same VM path (races → truncated file);
  serialize transfers and verify size before load.

### Serving gotchas (load-bearing)
- `pkill -f api_server` over SSH KILLS ITS OWN SHELL (pattern matches the command's own cmdline) →
  silent 255s. Kill by port (`fuser -k 8001/tcp`) or `pgrep -x python`, never `-f <pattern-in-cmd>`.
- Run vLLM under `tmux` (survives SSH disconnect); plain nohup/setsid over SSH got SIGHUP'd.
- vLLM `--lora-modules name=PATH` needs an EXPLICIT absolute path; a `find|dirname` that resolves
  empty yields `No adapter found for .` and the server exits.
- Image has python3 but NO pip → `apt-get install -y python3-pip`, use a venv.
- gpt-oss FT model id = `unsloth/gpt-oss-120b-BF16`; TF caps n_epochs ≤ 20; packing must be True.

## Path 2 (chosen): GPU-VM vLLM serving — turnkey post-auth

Scripts written + executable:
- `scripts/remote_bootstrap_vllm.sh` — runs ON the VM: pip-install vLLM, serve base
  `unsloth/gpt-oss-120b-BF16` + LoRA adapter, OpenAI-compatible on :8001 (TP = #GPUs).
- `scripts/nebius_serve_lora.sh <adapter_dir> <cache> <tag>` — local orchestrator: discover project →
  provision H200 VM → wait SSH → rsync adapter + bootstrap → launch vLLM → tunnel localhost:8001 →
  run base-vs-LoRA eval (`rollout` + `eval_selector` + `eval_adapter`) → print delete command.

ONE human gate (interactive, cannot automate):
```
! ~/.nebius/bin/nebius profile create     # browser/2FA federation auth
```
Then (turnkey):
```
export HF_TOKEN=<hf token for gpt-oss base weights>
scripts/nebius_serve_lora.sh adapters/gpt_oss_120b/hard_ftjob-d12e84aa9d5349ccb08404f7b17e3003 \
    gpt_oss/data/out/hard_cache.jsonl hard
```

Caveats / risks:
- gpt-oss-120b MoE bf16 ≈ 235 GB → needs ≥4× H200 (script defaults 8× preset). **~$30–40/hr** —
  delete the VM when done (script prints `nebius compute instance delete --id ...`).
- vLLM LoRA-on-gpt-oss-MoE is the main technical risk; the adapter targets attention proj modules,
  should load, but smoke the served base first (`/v1/models` → 200, one chat call) before trusting
  the adapter path. If vLLM rejects the MoE LoRA, fall back to merge (`merge-moe-lora-weights`) then
  serve the merged model with plain `--model`.
- SSH key `~/.ssh/id_ed25519.pub` present ✓. Nebius CLI 0.12.223 installed at `~/.nebius/bin/nebius`.

## Reproducibility / infra notes
- Inference base URL: `https://api.tokenfactory.nebius.com/v1/` (the `us-central1` host in old docs
  was wrong — purged).
- Inference model id `openai/gpt-oss-120b`; **fine-tune model id differs**: `unsloth/gpt-oss-120b-BF16`.
- gpt-oss is a REASONING model — spends completion tokens on hidden reasoning before content; small
  max_tokens → empty content. `client.chat()` default bumped 512→1024.
- hard_cache uses seeds {0,1,2}; build SFT/DPO with `--eval-seeds 1` (48 train / 24 eval).
