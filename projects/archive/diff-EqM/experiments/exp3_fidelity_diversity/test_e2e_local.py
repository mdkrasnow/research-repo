"""End-to-end CPU integration test for the Exp3 orchestrator with SYNTHETIC data.

Exercises the real Inception + resnet50 code paths, the cross-arm parity gate,
provenance-aware caching, CSV/JSON/plot outputs and the verdict -- all on a tiny
fabricated dataset. Metric VALUES are meaningless (12 random images); this tests
PLUMBING and the guards added during review.

Run: python projects/diff-EqM/experiments/exp3_fidelity_diversity/test_e2e_local.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ORCH = HERE / "eval_fidelity_diversity.py"


def write_pngs(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"{i:06d}.png")


def setup(tmp):
    tmp = Path(tmp)
    # schedule: 12 samples across 4 ImageNet class ids (num_classes=1000 for clf parity)
    labels = [10, 20, 30, 40] * 3
    sched = {"num_classes": 1000, "samples_per_class": 3, "num_samples": 12,
             "base_seed": 0, "shuffle_seed": 0,
             "labels": labels, "seeds": list(range(12))}
    run = tmp / "run"; run.mkdir(parents=True)
    (run / "schedule.json").write_text(json.dumps(sched))

    # fabricated FIXED reference cache (2048-d)
    ref = tmp / "reference"; ref.mkdir()
    rng = np.random.default_rng(1)
    agg = rng.normal(size=(60, 2048))
    np.save(ref / "inception_feats.npy", agg)
    np.savez(ref / "inception_mu_sigma.npz", mu=agg.mean(0), sigma=np.cov(agg, rowvar=False))
    np.save(ref / "real_classifier_hist.npy", np.ones(1000) / 1000)
    by = {}
    for c in (10, 20, 30, 40):
        cf = rng.normal(size=(10, 2048))
        by[f"mu_{c}"] = cf.mean(0); by[f"cov_{c}"] = np.cov(cf, rowvar=False)
        by[f"cnt_{c}"] = np.array(10)
    np.savez(ref / "inception_by_class.npz", **by)
    np.savez(ref / "inception_feats_by_class.npz",
             **{str(c): rng.normal(size=(10, 2048)).astype(np.float32)
                for c in (10, 20, 30, 40)})

    # gen folders with DIFFERENT provenance hashes + manifests
    for arm, h in [("vanilla", "VANIL0"), ("anm", "ANM001")]:
        g = run / "gen" / arm
        write_pngs(g, 12)
        (g / "gen_provenance.json").write_text(json.dumps({"hash": h}))
        with open(g / "manifest_part_0.csv", "w") as f:
            f.write("sample_id,seed,requested_label\n")
            for i in range(12):
                f.write(f"{i},{i},{labels[i]}\n")
    return run, ref


def run_orch(run, ref, extra=()):
    cmd = [sys.executable, str(ORCH),
           "--gen-root", str(run / "gen"), "--schedule", str(run / "schedule.json"),
           "--reference-dir", str(ref), "--out", str(run),
           "--vanilla-ckpt", "/fake/vanilla.pt", "--anm-ckpt", "/fake/anm.pt",
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run, ref = setup(tmp)

        # --- 1. happy path ---
        r = run_orch(run, ref)
        print("--- happy-path tail ---")
        print(r.stdout[-600:])
        assert r.returncode == 0, f"orchestrator failed:\n{r.stderr[-1500:]}"
        for fn in ["aggregate_metrics.csv", "aggregate_metrics.json",
                   "class_metrics.csv", "delta_class_metrics.csv",
                   "classifier_histogram.csv", "samples_manifest.csv", "README.md"]:
            assert (run / fn).exists(), f"missing output {fn}"
        plots = list((run / "plots").glob("*.png"))
        assert len(plots) >= 5, f"too few plots: {len(plots)}"
        summ = json.loads((run / "aggregate_metrics.json").read_text())
        assert summ["verdict"] in {"success", "strong_success", "failure", "ambiguous"}
        assert "schedule_hash" in summ
        print(f"[ok] happy path: verdict={summ['verdict']} plots={len(plots)} "
              f"sched={summ['schedule_hash']}")

        # --- 2. PARITY GATE: drop one anm sample ---
        (run / "gen" / "anm" / "000011.png").unlink()
        # bust caches so features re-extract on the now-11-image folder
        for p in run.glob("features_anm*"):
            p.unlink()
        for p in run.glob("classifier_anm*"):
            p.unlink()
        r2 = run_orch(run, ref)
        assert r2.returncode != 0, "parity gate did NOT fire on unequal sample sets"
        assert "PARITY FAIL" in (r2.stdout + r2.stderr), "missing PARITY FAIL message"
        print("[ok] parity gate fired on unequal sample-id sets")

        # restore
        write_pngs(run / "gen" / "anm", 12)

        # --- 3. provenance cache invalidation ---
        r3 = run_orch(run, ref)
        assert r3.returncode == 0, f"restore run failed:\n{r3.stderr[-800:]}"
        # change anm provenance -> next run must invalidate the anm feature cache
        (run / "gen" / "anm" / "gen_provenance.json").write_text(json.dumps({"hash": "ANM999"}))
        r4 = run_orch(run, ref)
        assert "invalidating" in r4.stdout, "feature cache NOT invalidated on provenance change"
        print("[ok] provenance change invalidated stale feature/pred cache")

        # --- 4. same-ckpt guard ---
        r5 = run_orch(run, ref, extra=["--anm-ckpt", "/fake/vanilla.pt"])
        assert r5.returncode != 0 and "PARITY FAIL" in (r5.stdout + r5.stderr), \
            "same-ckpt guard did not fire"
        print("[ok] same-checkpoint guard fired")

        print("\nALL E2E PLUMBING + GUARD CHECKS PASSED")


if __name__ == "__main__":
    main()
