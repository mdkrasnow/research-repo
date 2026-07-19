import json
import os
import torch
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

SP = os.path.dirname(os.path.abspath(__file__))
ASSETS = f"{SP}/../fourier_repl/severity_grid_assets"
selections = json.load(open(f"{SP}/selections_01.json"))
os.makedirs(f"{SP}/grids_convergence", exist_ok=True)

STEPS = [100, 250, 500, 1000]


def load_img(path):
    t = read_image(path).float() / 255.0
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t[:3]


for step in STEPS:
    for sel_name, indices in selections.items():
        rows = []
        for idx in indices:
            clean = load_img(f"{ASSETS}/clean_idx{idx}.png")
            corrupted = load_img(f"{ASSETS}/corrupted_cutoff0.1_idx{idx}.png")
            gaussian = load_img(f"{SP}/convergence_grid_gaussian_1_samples/cutoff0.1_step{step}_idx{idx}.png")
            mask = load_img(f"{SP}/convergence_grid_mask_1_samples/cutoff0.1_step{step}_idx{idx}.png")
            gm = load_img(f"{SP}/convergence_grid_gm_1_samples/cutoff0.1_step{step}_idx{idx}.png")
            rows.extend([clean, corrupted, gaussian, mask, gm])
        grid = make_grid(torch.stack(rows), nrow=5, padding=2)
        out_path = f"{SP}/grids_convergence/grid_step{step}_{sel_name}.png"
        save_image(grid, out_path)
        print("wrote", out_path)
