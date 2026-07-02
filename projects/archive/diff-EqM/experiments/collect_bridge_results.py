#!/usr/bin/env python3
"""Collect v13 SE2 bridge results into a compact markdown table + decision read.

Robust to missing/partial jobs. Runs on the CLUSTER (reads results dirs + slurm logs). For each arm:
parses final FID from the slurm log (`cifar10_variant_fid[...]: <fid>`), reads operator_diag.json from the
run dir, and sacct state. Reused arms (v00, vK_rotation) carry their already-measured FID.

Usage (on cluster, via scripts/cluster/ssh.sh):
    python projects/diff-EqM/experiments/collect_bridge_results.py
    python projects/diff-EqM/experiments/collect_bridge_results.py --append-tsv
"""
import argparse, glob, json, os, re, subprocess

RR = "/n/home03/mkrasnow/research-repo"
RESULTS = f"{RR}/projects/diff-EqM/results"
LOGS = f"{RR}/slurm/logs"   # variant_pilot #SBATCH --output is submit-dir-relative -> $RR/slurm/logs

# (arm, variant_name, job_id, role, reused_fid)
ARMS = [
    ("v00_base",            "v00_vanilla",          "19242268", "floor (reused)",            14.31),
    ("vK_rotation",         "vK_known_aug",         "19234360", "wrong-transform ctrl (reused)", 14.82),
    ("v10_hardneg",         "v10_hard_example",     "19405301", "mining competitor",         None),
    ("vK_translate_crop",   "vK_known_aug",         "19404840", "POSITIVE control",          None),
    ("v13_discovered_se2",  "v13_stable_se2_aug",   "19404844", "TREATMENT",                 None),
    ("v13_random_se2",      "v13_stable_se2_aug",   "19404851", "negative control",          None),
]


def sacct_state(job):
    try:
        out = subprocess.run(["sacct", "-j", job, "--format=State", "-n"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            s = line.strip()
            if s and "." not in line.split()[0]:
                return s.split()[0]
    except Exception:
        pass
    return "?"


def parse_fid(job):
    # final FID line: "cifar10_variant_fid[<variant>]: <fid>"
    for f in sorted(glob.glob(f"{LOGS}/variant-pilot_{job}_*.out"), key=os.path.getmtime, reverse=True):
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        m = re.findall(r"cifar10_variant_fid\[[^\]]*\]:\s*([0-9.]+)", txt)
        if m:
            return float(m[-1])
        # fall back: latest epoch eval FID
        m2 = re.findall(r"FID\(ema[^)]*\)\s*=\s*([0-9.]+)", txt)
        if m2:
            return float(m2[-1])
    return None


def latest_epoch(job):
    for f in sorted(glob.glob(f"{LOGS}/variant-pilot_{job}_*.out"), key=os.path.getmtime, reverse=True):
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        m = re.findall(r"epoch (\d+)/(\d+)", txt)
        if m:
            return f"{m[-1][0]}/{m[-1][1]}"
    return "-"


def operator_diag(variant, job):
    d = f"{RESULTS}/variant_{variant}_{job}_seed0/operator_diag.json"
    if os.path.isfile(d):
        try:
            return json.load(open(d))
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append-tsv", action="store_true")
    args = ap.parse_args()

    rows = []
    print("\n## v13 SE2 bridge results (auto-collected)\n")
    print("| arm | role | state | epoch | FID | op:tx_px | det | cond | anchor b→a |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, variant, job, role, reused in ARMS:
        state = "DONE(reused)" if reused is not None else sacct_state(job)
        fid = reused if reused is not None else parse_fid(job)
        ep = "-" if reused is not None else latest_epoch(job)
        diag = operator_diag(variant, job) if reused is None else None
        tx = det = cond = anch = "-"
        if diag:
            tx = f"{diag.get('tx_px','-')}" if isinstance(diag.get('tx_px'), (int, float)) else "-"
            det = f"{diag.get('det','-'):.3f}" if isinstance(diag.get('det'), (int, float)) else "-"
            cond = f"{diag.get('cond','-'):.3f}" if isinstance(diag.get('cond'), (int, float)) else "-"
            b, a = diag.get('anchor_baseline_real_real'), diag.get('anchor_final_T_real')
            anch = f"{b:.2f}→{a:.2f}" if isinstance(b, (int, float)) and isinstance(a, (int, float)) else "-"
        fid_s = f"{fid:.3f}" if isinstance(fid, (int, float)) else "(pending)"
        if isinstance(tx, float):
            tx = f"{tx:.2f}"
        print(f"| {arm} | {role} | {state} | {ep} | {fid_s} | {tx} | {det} | {cond} | {anch} |")
        rows.append((arm, variant, job, role, fid, diag))

    # decision read (only if the key arms have FIDs)
    fids = {a: f for a, v, j, r, f, d in rows}
    def have(*ks): return all(isinstance(fids.get(k), (int, float)) for k in ks)
    print("\n### Decision read")
    if not have("vK_translate_crop", "v00_base"):
        print("- waiting on positive control (vK_translate_crop) + base")
    else:
        base = fids["v00_base"]; tc = fids["vK_translate_crop"]
        if tc >= base:
            print(f"- **A. HARNESS FAIL**: translate_crop {tc:.2f} ≥ base {base:.2f} → do NOT interpret v13.")
        elif have("v13_discovered_se2", "v13_random_se2"):
            disc = fids["v13_discovered_se2"]; rnd = fids["v13_random_se2"]
            v10 = fids.get("v10_hardneg")
            if rnd < disc:
                print(f"- **B. GENERIC REG**: random {rnd:.2f} < discovered {disc:.2f} → gain not from discovery.")
            elif disc < base and disc < rnd and isinstance(v10, (int, float)) and disc < v10:
                print(f"- **D. STRONG BRIDGE WIN**: discovered {disc:.2f} < base/random/v10 ({base:.2f}/{rnd:.2f}/{v10:.2f}).")
            elif disc < base and disc < rnd:
                print(f"- **C. WORKS, NOT > v10**: discovered {disc:.2f} < base/random; v10={v10}.")
            elif tc < base:
                print(f"- **E. KNOWN WINS, v13 DOESN'T**: translate_crop {tc:.2f} < base, discovered {disc:.2f} not.")
            else:
                print("- **F. INCONCLUSIVE / within noise** — controls do not separate.")
        else:
            print("- positive control valid; waiting on v13 treatment + random arms.")

    if args.append_tsv:
        tsv = f"{RR}/projects/diff-EqM/results_variants.tsv"
        with open(tsv, "a") as fh:
            for arm, variant, job, role, fid, diag in rows:
                if isinstance(fid, (int, float)) and "reused" not in role:
                    fh.write(f"bridge150_se2\t{arm}\t seed0\t{fid:.4f}\tCOLLECTED\t{role}\tjob{job}\t2026-06-05\n")
        print(f"\nappended completed non-reused rows to {tsv}")


if __name__ == "__main__":
    main()
