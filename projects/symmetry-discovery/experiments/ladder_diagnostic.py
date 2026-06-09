"""ladder_diagnostic — interpret the Track-B v10-comparison ladder once it lands.

Reads each arm's final FID + (for policy arms) operator_diag.json, and emits:
  1. the staged-success verdict (min/strong/major/best per v16_v10_comparison_ladder.md),
  2. an interpretable readout of WHAT the discovered policy picked on full CIFAR
     (spatial vs photometric vs decoy mass, per-family weight x magnitude),
  3. the pre-registered diagnostic: if discovery loses to v10, WHY — weak-but-valid families?
     wrong magnitude? under-weighting spatial transforms? over-favoring hue/bright?

Usage (after fetching results, or pointed at the cluster persist root):
  python ladder_diagnostic.py --fids v00=14.31,v10=13.40,v17rand=14.59,v17disc=14.21,hybrid=13.1 \
      --diag-dir projects/diff-EqM/results/variants
  # diag JSONs expected at <diag-dir>/<...v17disc...>/operator_diag.json etc, or pass --diag-map.

FID values: pass measured numbers via --fids (run reads cluster train.log `cifar10_dganm_fid`). The diag
dir is scanned for operator_diag.json files; match by arm substring.
"""
from __future__ import annotations
import argparse, glob, json, os

SPATIAL = {"translate_x", "translate_y", "rotate", "scale"}
PHOTO = {"hue", "bright", "saturate"}
DECOY = {"crop_erase", "big_shear", "color_collapse"}


def parse_fids(s):
    out = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def find_diag(diag_dir, arm):
    # match a single operator_diag.json whose path contains the arm tag
    hits = [p for p in glob.glob(os.path.join(diag_dir, "**", "operator_diag.json"), recursive=True)
            if arm in p]
    return hits[0] if hits else None


def mass(weights, keys):
    return sum(v for k, v in weights.items() if k in keys)


def policy_readout(diag):
    w = diag.get("family_weights") or {}
    mag = diag.get("mag_mu") or {}
    eff = diag.get("effective_usage") or {}
    decoy = diag.get("decoy_usage", mass(w, DECOY))
    top = sorted(w.items(), key=lambda kv: -kv[1])[:5]
    return {
        "decoy_usage": round(decoy, 4),
        "spatial_mass": round(mass(w, SPATIAL), 3),
        "photo_mass": round(mass(w, PHOTO), 3),
        "decoy_mass": round(mass(w, DECOY), 3),
        "top5": [(k, round(v, 3), round(abs(mag.get(k, 0.0)), 3)) for k, v in top],  # (family, weight, |mag|)
        "effective_top": sorted(eff.items(), key=lambda kv: -kv[1])[:5] if eff else None,
    }


def diagnose(fids, readouts):
    """Pre-registered staged verdict + why-if-lose."""
    lines = []
    base = fids.get("v00"); v10 = fids.get("v10")
    rnd = fids.get("v17rand"); disc = fids.get("v17disc"); hyb = fids.get("hybrid")

    def cmp(a, b):
        return "?" if a is None or b is None else ("<" if a < b else (">" if a > b else "="))

    lines.append("## Staged success (lower FID better)")
    if disc is not None and rnd is not None:
        lines.append(f"- MIN  (disc<random):  v17disc {disc} {cmp(disc,rnd)} v17rand {rnd}  "
                     f"-> {'PASS' if disc<rnd else 'FAIL'}")
    if disc is not None and base is not None:
        lines.append(f"- STRONG(disc<base):   v17disc {disc} {cmp(disc,base)} v00 {base}  "
                     f"-> {'PASS' if disc<base else 'FAIL'}")
    if disc is not None and v10 is not None:
        lines.append(f"- MAJOR (disc<v10):    v17disc {disc} {cmp(disc,v10)} v10 {v10}  "
                     f"-> {'PASS' if disc<v10 else 'FAIL'}")
    if hyb is not None and v10 is not None:
        lines.append(f"- BEST  (hybrid<v10):  hybrid {hyb} {cmp(hyb,v10)} v10 {v10}  "
                     f"-> {'PASS' if hyb<v10 else 'FAIL'}")

    # why-if-lose diagnostic on v17disc policy
    rd = readouts.get("v17disc")
    if rd and disc is not None and v10 is not None and disc >= v10:
        lines.append("\n## Diagnostic — v17disc did NOT beat v10. Likely cause:")
        if rd["decoy_usage"] > 0.1:
            lines.append(f"- DECOY LEAK: decoy_usage {rd['decoy_usage']} > 0.1 — destructive aug poisoning.")
        if rd["photo_mass"] > rd["spatial_mass"] + 0.2:
            lines.append(f"- PHOTOMETRIC-HEAVY: photo {rd['photo_mass']} >> spatial {rd['spatial_mass']}. "
                         f"On full CIFAR the high-value augs are SPATIAL (crop/translate); over-weighting "
                         f"hue/bright/saturate = valid-but-low-value -> underperforms crop-like v10.")
        if rd["spatial_mass"] > rd["photo_mass"] + 0.2:
            lines.append(f"- SPATIAL-HEAVY but still lost: spatial {rd['spatial_mass']}. Check MAGNITUDE — "
                         f"weak translate/scale (small |mag|) ~ identity; strong ~ crop. See top5 |mag|.")
        mags = [m for _, _, m in rd["top5"]]
        if mags and max(mags) < 0.15:
            lines.append(f"- WEAK MAGNITUDE: top |mag| {max(mags)} < 0.15 — policy near-identity, little aug.")
        lines.append(f"- policy: {rd}")
    elif rd:
        lines.append(f"\n## v17disc policy readout: {rd}")
    rh = readouts.get("hybrid")
    if rh:
        lines.append(f"## hybrid policy readout: {rh}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fids", required=True, help="v00=..,v10=..,v17rand=..,v17disc=..,hybrid=..")
    ap.add_argument("--diag-dir", default="projects/diff-EqM/results/variants")
    ap.add_argument("--diag-map", default="", help="optional arm=path,.. override for operator_diag.json")
    args = ap.parse_args()

    fids = parse_fids(args.fids)
    dmap = {}
    if args.diag_map:
        for kv in args.diag_map.split(","):
            k, v = kv.split("="); dmap[k.strip()] = v.strip()

    readouts = {}
    for arm in ("v17disc", "hybrid"):
        path = dmap.get(arm) or find_diag(args.diag_dir, arm)
        if path and os.path.exists(path):
            readouts[arm] = policy_readout(json.load(open(path)))
        else:
            print(f"[warn] no operator_diag for {arm} (looked: {path})")

    print("# Track-B v10 ladder — diagnostic\n")
    print("## FIDs (lower better)")
    for k in ("v00", "v10", "v17rand", "v17disc", "hybrid"):
        if k in fids:
            print(f"- {k}: {fids[k]}")
    print()
    print(diagnose(fids, readouts))


if __name__ == "__main__":
    main()
