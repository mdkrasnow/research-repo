"""
Canonical measured trace/action/outcome builder (the dataset SPINE).

Reuses the Phase-3 machinery (experiments/phase3.py) to produce, for each (seed, failure-mode)
episode, by REAL reruns only:
  - observable_trace : label-free signals a selector legitimately sees (mechanism probe + oracle
                       sandwich). NO ground-truth action, NO hidden score.
  - reward_by_action : measured true-score after actually applying each lever
                       {H, W, H_THEN_W, NOOP}. Real training reruns, not a transition table.
  - best_action      : argmax over the three active levers (H/W/H_THEN_W).
  - correct          : the pre-registered correct lever for the mode (for auditing).

Outputs (under gpt_oss/data/out/):
  action_outcome_cache.jsonl   one row per episode; THE measured cache everything else consumes.

This is the only place that runs models. SFT/DPO/GRPO builders and the SIA task private data are
all derived from action_outcome_cache.jsonl — so there is a single measured source of truth.

NOOP/PROMOTE/KILL note: the three failure modes are all "something is broken, intervene". NOOP
(= keep the current model, what PROMOTE implies) is measured (true-score of the starting model).
KILL (abandon a salvageable mechanism) is assigned a documented floor of 0.0. Both are therefore
always-suboptimal here by construction of the modes, not by a hand-authored delta.

Run:  python gpt_oss/data/build_trace_dataset.py --seeds 10 --steps 800
"""

import argparse
import hashlib
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "experiments"))

import phase3  # noqa: E402  (make_episode, apply_lever, true_score, mechanism_probe, reference_model, harness_score, LEVERS)

MODES = ["shortcut_leak", "model_prior_gap", "bad_verifier"]
KILL_FLOOR = 0.0  # documented: abandoning a salvageable episode yields zero useful model


def observable_trace(ep, seed):
    """Label-free signals a selector may legitimately compute (same info selector_rule uses)."""
    ref = phase3.reference_model()
    ref_h = phase3.harness_score(ep["harness"], ref, seed=seed)
    harness_valid = ref_h < 0.05                      # does the deployed harness accept a known-good model?
    probe = phase3.mechanism_probe(ep["model"], seed=seed)
    return {
        "clean_mse": round(probe["clean_mse"], 6),
        "neg_control_mse": round(probe["neg_control_mse"], 6),
        "composition_error": round(probe["composition_error"], 6),
        "predicts_clean": bool(probe["predicts_clean"]),
        "solves_broken_symmetry": bool(probe["neg_control_mse"] < 0.25),
        "shortcut_cheat_signature": bool(probe["cheats"]),
        "reference_model_score_under_harness": round(ref_h, 6),
        "harness_accepts_known_good_model": bool(harness_valid),
    }


def render_trace_text(obs):
    """Human/LLM-readable rendering of the observable trace."""
    return (
        "FAILED-RUN TRACE (deployed harness + model under test)\n"
        f"- clean prediction MSE: {obs['clean_mse']:.4f} "
        f"({'predicts clean examples well' if obs['predicts_clean'] else 'cannot predict clean examples'})\n"
        f"- negative-control MSE: {obs['neg_control_mse']:.4f} "
        f"({'SOLVES the broken-symmetry control (suspicious)' if obs['solves_broken_symmetry'] else 'fails the broken-symmetry control (expected for an honest model)'})\n"
        f"- learned-action composition error: {obs['composition_error']:.4f} "
        f"({'group law violated' if obs['composition_error'] > 0.10 else 'group law ~satisfied'})\n"
        f"- shortcut-cheat signature present: {obs['shortcut_cheat_signature']}\n"
        f"- ORACLE SANDWICH: a known-good reference model scores {obs['reference_model_score_under_harness']:.4f} "
        f"under the deployed harness -> harness {'ACCEPTS' if obs['harness_accepts_known_good_model'] else 'REJECTS'} a known-good model "
        f"({'harness appears valid' if obs['harness_accepts_known_good_model'] else 'harness appears BROKEN'}).\n"
    )


def build(seeds, steps):
    rows = []
    for seed in range(seeds):
        for idx, mode in enumerate(MODES):
            ep = phase3.make_episode(mode, steps=steps, seed=seed * 10 + idx)
            obs = observable_trace(ep, seed)
            reward = {}
            for lever in phase3.LEVERS:                    # H, W, H_THEN_W -> real reruns
                m = phase3.apply_lever(lever, ep, steps=steps, seed=seed)
                reward[lever] = round(phase3.true_score(m, seed=seed), 6)
            reward["NOOP"] = round(phase3.true_score(ep["model"], seed=seed), 6)   # measured no-op
            reward["KILL"] = KILL_FLOOR                                            # documented floor
            best_action = max(phase3.LEVERS, key=lambda k: reward[k])
            rows.append({
                "episode_id": f"{mode}_seed_{seed:03d}",
                "mode": mode,
                "seed": seed,
                "observable_trace": obs,
                "trace_text": render_trace_text(obs),
                "reward_by_action": reward,
                "best_action": best_action,
                "correct_action": ep["correct"],
            })
            print(f"{mode}_seed_{seed:03d}: best={best_action} correct={ep['correct']} "
                  f"reward={ {k: round(v,3) for k,v in reward.items()} }", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(PROJ, "gpt_oss", "data", "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = build(args.seeds, args.steps)
    cache_path = os.path.join(args.out, "action_outcome_cache.jsonl")
    with open(cache_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    blob = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    meta = {"n_episodes": len(rows), "seeds": args.seeds, "steps": args.steps,
            "modes": MODES, "actions": ["H", "W", "H_THEN_W", "NOOP", "KILL"],
            "dataset_sha256_16": digest,
            "source": "real reruns via experiments/phase3.py (no transition table)"}
    with open(os.path.join(args.out, "action_outcome_cache.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {len(rows)} episodes -> {cache_path}\nhash {digest}")


if __name__ == "__main__":
    main()
