# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Qualitative grids for the G+M -> Fourier replication (2026-07-15 runbook).
Consumes per-image eval_fourier_recovery.py outputs (which must have been
run with --save-images so recovered PNGs exist on disk) for: gaussian,
mask, gm, fourier-specialist, plus one clean/corrupted reference pair, and
assembles four grids without hand selection:
  - 16 fixed-random images (first 16 manifest indices, deterministic)
  - 16 largest G+M wins (synergy = min(gaussian,mask) - gm, most positive)
  - 16 largest G+M losses (synergy most negative)
  - 16 images nearest the median synergy value
Also emits a blinded randomized A/B sheet: for each of the 16 fixed-random
images, the 4 recovery columns (gaussian/mask/gm/fourier-specialist) in a
per-image random order, with an answer key saved separately (not shown in
the sheet itself) so a human reviewer is not biased by column identity.
"""
import argparse
import json
import os

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image


def load_lpips_lookup(path, cutoff_key):
    with open(path) as f:
        d = json.load(f)
    pc = d["per_cutoff"][cutoff_key]
    return {r["index"]: r for r in pc["per_image"]}


def img_tensor(path):
    return read_image(path).float() / 255.0


def build_grid(indices, columns, out_path, label_row_height=0):
    """columns: dict[name] -> dict[index] -> png path. Missing entries
    skipped (row omitted) rather than erroring, since not every arm may
    have run every image (should not happen in a clean run, but don't
    crash a report over one missing sample)."""
    rows = []
    for idx in indices:
        row_imgs = []
        ok = True
        for name, lookup in columns.items():
            rec = lookup.get(idx)
            if rec is None or rec.get("out_path") is None or not os.path.exists(rec["out_path"]):
                ok = False
                break
            row_imgs.append(img_tensor(rec["out_path"]))
        if ok:
            rows.extend(row_imgs)
    if not rows:
        print(f"WARNING: no complete rows for {out_path}, skipping")
        return
    grid = make_grid(torch.stack(rows), nrow=len(columns))
    save_image(grid, out_path)
    print(f"-> {out_path} ({len(rows) // len(columns)} rows x {len(columns)} cols)")


def main(args):
    cutoff = args.cutoff
    gaussian = load_lpips_lookup(args.gaussian_json, cutoff)
    mask = load_lpips_lookup(args.mask_json, cutoff)
    gm = load_lpips_lookup(args.gm_json, cutoff)
    fourier_spec = load_lpips_lookup(args.fourier_specialist_json, cutoff) if args.fourier_specialist_json else {}

    common = sorted(set(gaussian) & set(mask) & set(gm))
    synergy = {}
    for idx in common:
        g_lp, m_lp, gm_lp = gaussian[idx]["lpips"], mask[idx]["lpips"], gm[idx]["lpips"]
        if None in (g_lp, m_lp, gm_lp):
            continue
        synergy[idx] = min(g_lp, m_lp) - gm_lp

    sorted_by_synergy = sorted(synergy, key=lambda i: synergy[i])
    n = args.grid_size

    fixed_random = common[:n]
    largest_wins = sorted_by_synergy[-n:][::-1]
    largest_losses = sorted_by_synergy[:n]
    median_val = np.median(list(synergy.values()))
    nearest_median = sorted(synergy, key=lambda i: abs(synergy[i] - median_val))[:n]

    columns = {"gaussian": gaussian, "mask": mask, "gm": gm}
    if fourier_spec:
        columns["fourier_specialist"] = fourier_spec

    os.makedirs(args.out_dir, exist_ok=True)
    build_grid(fixed_random, columns, os.path.join(args.out_dir, "grid_fixed_random.png"))
    build_grid(largest_wins, columns, os.path.join(args.out_dir, "grid_largest_wins.png"))
    build_grid(largest_losses, columns, os.path.join(args.out_dir, "grid_largest_losses.png"))
    build_grid(nearest_median, columns, os.path.join(args.out_dir, "grid_nearest_median.png"))

    with open(os.path.join(args.out_dir, "selection_indices.json"), "w") as f:
        json.dump({
            "fixed_random": fixed_random, "largest_wins": largest_wins,
            "largest_losses": largest_losses, "nearest_median": nearest_median,
            "median_synergy": float(median_val),
            "synergy_by_index": {str(k): float(v) for k, v in synergy.items()},
        }, f, indent=2)

    # blinded A/B sheet: per-image random column order, answer key kept separate
    rng = np.random.default_rng(args.rng_seed)
    col_names = list(columns.keys())
    ab_sheet_rows = []
    answer_key = []
    for idx in fixed_random:
        order = rng.permutation(len(col_names)).tolist()
        shuffled_names = [col_names[o] for o in order]
        row_imgs = [img_tensor(columns[name][idx]["out_path"]) for name in shuffled_names]
        ab_sheet_rows.extend(row_imgs)
        answer_key.append({"index": idx, "column_order": shuffled_names})
    if ab_sheet_rows:
        grid = make_grid(torch.stack(ab_sheet_rows), nrow=len(col_names))
        save_image(grid, os.path.join(args.out_dir, "ab_sheet_blinded.png"))
        with open(os.path.join(args.out_dir, "ab_sheet_answer_key.json"), "w") as f:
            json.dump(answer_key, f, indent=2)
        print(f"-> {os.path.join(args.out_dir, 'ab_sheet_blinded.png')} "
              f"(answer key kept separate: ab_sheet_answer_key.json)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-json", type=str, required=True)
    parser.add_argument("--mask-json", type=str, required=True)
    parser.add_argument("--gm-json", type=str, required=True)
    parser.add_argument("--fourier-specialist-json", type=str, default=None)
    parser.add_argument("--cutoff", type=str, default="0.4181")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
