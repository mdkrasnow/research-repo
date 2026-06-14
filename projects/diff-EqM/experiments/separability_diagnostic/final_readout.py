"""Task 5 — final readout generator (cached-only, one report).

Given a completed run dir, stitches a single decision-grade markdown report:
  verdict -> baseline table -> rejection-payoff -> controls -> next action.

Next-action rule (on the de-confounded within-norm verdict):
  GREEN  => run probe-guided rejection sampler (fid_payoff.py / probe_gated_sample.py)
  WEAK   => run learned trajectory probe (dynamics_probe.py / learned_probe.py)
  KILL   => pivot from scalar to trajectory-shape / vector features

Auto-runs baseline_table.py and rejection_payoff.py if their outputs are absent.
Fails gracefully, naming the missing path + the exact command to produce it.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd):
    print(f"[final_readout] running: {cmd}")
    return subprocess.run(cmd, shell=True).returncode


def ensure(folder, rel, gen_cmd):
    p = Path(folder) / rel
    if not p.exists():
        sh(gen_cmd)
    return p if p.exists() else None


def read_verdict(folder):
    for name in ("VERDICT.txt", "PROBE_VERDICT.txt"):
        p = Path(folder) / "results" / name
        if p.exists():
            return p.read_text().strip()
    return None


def best_within(folder):
    """Best de-confounded within-norm AUROC across scalars + learned probe."""
    bt = Path(folder) / "results" / "baseline_table.csv"
    best, best_m = None, None
    if bt.exists():
        with open(bt) as fh:
            for r in csv.DictReader(fh):
                if r["method"].startswith("latent-NN") or r["method"] == "random":
                    continue
                try:
                    w = float(r["within_norm_auroc"])
                except ValueError:
                    continue
                if w == w and (best is None or w > best):
                    best, best_m = w, r["method"]
    return best, best_m


def next_action(within):
    if within is None:
        return "INCONCLUSIVE", "Re-run the diagnostic / probe; no de-confounded AUROC available."
    if within >= 0.80:
        return ("GREEN", "Run probe-guided rejection sampler: `fid_payoff.py` (pool reject) then "
                         "`probe_gated_sample.py` / `online_adaptive_sampler.py` (in-line restart).")
    if within >= 0.60:
        return ("WEAK", "Run the learned trajectory probe: `dynamics_probe.py` + `learned_probe.py` "
                        "(full descent-shape features, held-out, shuffle control).")
    return ("KILL", "Pivot from scalar readouts to trajectory-shape / vector features "
                    "(per-step curve geometry, not endpoint scalars).")


def main(a):
    folder = Path(a.folder)
    if not folder.exists():
        print(f"[final_readout] MISSING: {folder}\n[next] point --folder at a completed run dir.")
        sys.exit(2)

    bt = ensure(folder, "results/baseline_table.csv",
                f"python {HERE/'baseline_table.py'} --folder {folder}")
    rp = ensure(folder, "results/REJECTION_PAYOFF.md",
                f"python {HERE/'rejection_payoff.py'} --folder {folder}")

    verdict = read_verdict(folder)
    within, within_m = best_within(folder)
    tag, action = next_action(within)

    out = ["# EqM Separability — Final Readout", "",
           f"Run dir: `{folder}`", ""]
    out += ["## Verdict", "```", verdict or "(no VERDICT.txt found)", "```", ""]
    out += [f"## Decision: **{tag}**",
            f"Best de-confounded within-norm AUROC: "
            f"{'%.3f' % within if within is not None else 'n/a'}"
            f"{f' ({within_m})' if within_m else ''}.", "",
            f"**Next action:** {action}", ""]

    btmd = folder / "results" / "BASELINE_TABLE.md"
    if btmd.exists():
        out += ["## Baseline table", btmd.read_text().split("\n", 2)[-1], ""]
    elif bt:
        out += ["## Baseline table", f"(see `{bt}`)", ""]

    if rp and rp.exists():
        out += ["## Rejection-payoff", rp.read_text().split("\n", 2)[-1], ""]

    out += ["## Controls present",
            "- rejection: random, norm_only, shuffled-score, shuffled-label",
            "- AUROC: raw vs within-norm (matched-norm de-confound), latent-NN s4 sanity",
            "- probe (if run): held-out split, fixed seed, label-shuffle", ""]

    rep = folder / "results" / "FINAL_READOUT.md"
    rep.write_text("\n".join(out) + "\n")
    print(f"[final_readout] wrote {rep}")
    print("\n".join(out[:18]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="completed run dir")
    main(ap.parse_args())
