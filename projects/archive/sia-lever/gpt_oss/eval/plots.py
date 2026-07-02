"""Shared demo-quality plotting + diagnostics helpers for the gpt-oss lever experiments.

Every eval writes: a figure (PNG), a human markdown table, and a per-episode diagnostics file that
lists mistakes (predicted != correct) with the model's raw response — so a failure is debuggable at
a glance. Pure matplotlib; safe to import on CPU.
"""

import json
import os

MODES = ["shortcut_leak", "model_prior_gap", "bad_verifier"]
MODE_SHORT = {"shortcut_leak": "shortcut", "model_prior_gap": "weak-model", "bad_verifier": "bad-harness"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def bar_labels(ax, bars, fmt="{:.2f}"):
    """Annotate vertical bars with their height (all callers here use vertical bars)."""
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8)


def per_mode_bar(per_mode_acc, title, path, ref=None):
    """Grouped bar of per-mode lever accuracy. ref optional dict for a second series (e.g. base)."""
    plt = _mpl()
    modes = [m for m in MODES if m in per_mode_acc] or list(per_mode_acc)
    xs = range(len(modes))
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [per_mode_acc[m] for m in modes]
    if ref:
        w = 0.38
        b1 = ax.bar([x - w / 2 for x in xs], [ref.get(m, 0) for m in modes], w, label="base", color="tab:gray")
        b2 = ax.bar([x + w / 2 for x in xs], vals, w, label="model", color="tab:green")
        bar_labels(ax, list(b1) + list(b2)); ax.legend()
    else:
        bar_labels(ax, ax.bar(list(xs), vals, color="tab:green", width=0.6))
    ax.set_xticks(list(xs)); ax.set_xticklabels([MODE_SHORT.get(m, m) for m in modes])
    ax.set_ylim(0, 1.05); ax.set_ylabel("lever accuracy"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def base_vs_adapter_fig(base, adapter, path, tag="LoRA"):
    """Headline before/after figure: lever_acc (up=good), mean_regret (down=good), invalid_json (down)."""
    plt = _mpl()
    metrics = [("lever_accuracy", "lever acc ↑", False),
               ("mean_regret", "mean regret ↓", True),
               ("invalid_json_rate", "invalid JSON ↓", True)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (k, label, lower_better) in zip(axes, metrics):
        b = base.get(k) if base else None
        a = adapter.get(k)
        names = (["base", f"+{tag}"] if base else [f"+{tag}"])
        vals = ([b, a] if base else [a])
        colors = (["tab:gray", "tab:green"] if base else ["tab:green"])
        bars = ax.bar(names, vals, color=colors, width=0.6)
        bar_labels(ax, bars, "{:.3f}")
        ax.set_title(label)
        if k != "mean_regret":
            ax.set_ylim(0, 1.05)
        if base is not None:
            d = a - b
            good = (d < 0) if lower_better else (d > 0)
            ax.annotate(f"Δ {d:+.3f}", (0.5, 0.92), xycoords="axes fraction", ha="center",
                        color=("green" if good else "red"), fontsize=10, weight="bold")
    fig.suptitle(f"gpt-oss-120b base vs base+{tag} — lever selection")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def action_distribution_fig(dist, title, path):
    plt = _mpl()
    items = [(k, v) for k, v in dist.items() if v]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bar_labels(ax, ax.bar([k for k, _ in items], [v for _, v in items], color="tab:blue"), "{:.0f}")
    ax.set_title(title); ax.set_ylabel("count"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_diagnostics(rollouts, cache, out_md, out_jsonl, gold_fn, outcome_fn):
    """Per-episode diagnostics + a MISTAKES section. gold_fn(rba)->action, outcome_fn(action,rba)->score."""
    by_id = {r["episode_id"]: r for r in cache}
    rows, mistakes = [], []
    for r in rollouts:
        ep = by_id.get(r["episode_id"])
        if not ep:
            continue
        rba = ep["reward_by_action"]
        gold = gold_fn(rba)
        act = r.get("action")
        regret = outcome_fn(gold, rba) - outcome_fn(act if act else "KILL", rba)
        row = {"episode_id": r["episode_id"], "mode": ep["mode"], "correct": gold,
               "predicted": act, "valid_json": r.get("valid_json"), "regret": round(regret, 4),
               "reason": (r.get("reason") or "")[:160],
               "raw_snippet": (str(r.get("raw_response") or ""))[:200]}
        rows.append(row)
        if act != gold:
            mistakes.append(row)
    with open(out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    md = [f"# Diagnostics ({len(rows)} episodes, {len(mistakes)} mistakes)", "",
          "## Mistakes (predicted != correct)", ""]
    if not mistakes:
        md.append("_none — perfect lever accuracy on this set._")
    for m in mistakes:
        md += [f"- **{m['episode_id']}** ({m['mode']}): correct=`{m['correct']}` "
               f"predicted=`{m['predicted']}` regret={m['regret']} valid_json={m['valid_json']}",
               f"  - reason: {m['reason']}",
               f"  - raw: `{m['raw_snippet']}`"]
    md += ["", "## All episodes", "",
           "| episode | mode | correct | predicted | regret | valid_json |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['episode_id']} | {r['mode']} | {r['correct']} | {r['predicted']} "
                  f"| {r['regret']} | {r['valid_json']} |")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    return len(mistakes)
