import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(f"{SP}/convergence_analysis.json"))
steps = [int(s) for s in d["steps"]]
r = d["results"]

# 1. LPIPS vs recovery steps, all models
plt.figure(figsize=(7, 5))
plt.plot(steps, [r[str(s)]["mean_gaussian"] for s in steps], "o-", label="Gaussian-only")
plt.plot(steps, [r[str(s)]["mean_mask"] for s in steps], "s-", label="Mask-only")
plt.plot(steps, [r[str(s)]["mean_gm"] for s in steps], "^-", label="Gaussian+mask 1:1")
plt.xscale("symlog", linthresh=25)
plt.xlabel("Recovery step")
plt.ylabel("Mean LPIPS (lower = better)")
plt.title("LPIPS vs recovery steps (fourier cutoff 0.10)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/conv_plot1_lpips_vs_steps.png", dpi=150)
plt.close()

# 2. delta_G(t), delta_M(t) vs steps with CI bands
plt.figure(figsize=(7, 5))
dg = [r[str(s)]["delta_g"] for s in steps]
dg_lo = [r[str(s)]["ci_g"][0] for s in steps]
dg_hi = [r[str(s)]["ci_g"][1] for s in steps]
dm = [r[str(s)]["delta_m"] for s in steps]
dm_lo = [r[str(s)]["ci_m"][0] for s in steps]
dm_hi = [r[str(s)]["ci_m"][1] for s in steps]
plt.plot(steps, dg, "o-", color="C0", label="delta_G (gaussian - GM)")
plt.fill_between(steps, dg_lo, dg_hi, color="C0", alpha=0.2)
plt.plot(steps, dm, "s-", color="C1", label="delta_M (mask - GM)")
plt.fill_between(steps, dm_lo, dm_hi, color="C1", alpha=0.2)
plt.axhline(0, color="gray", linewidth=1)
plt.axvline(250, color="red", linestyle=":", linewidth=1, label="original 250-step horizon")
plt.xscale("symlog", linthresh=25)
plt.xlabel("Recovery step")
plt.ylabel("LPIPS delta (positive = GM wins)")
plt.title("delta_G(t) / delta_M(t) vs recovery steps (bootstrap 95% CI)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/conv_plot2_deltas_vs_steps.png", dpi=150)
plt.close()

# 3. seed-level delta_G(t) trajectories
plt.figure(figsize=(7, 5))
for seed in range(5):
    traj = [r[str(s)]["seed_means_gaussian"][seed] - r[str(s)]["seed_means_gm"][seed] for s in steps]
    plt.plot(steps, traj, "o-", alpha=0.7, label=f"seed{seed}")
plt.axhline(0, color="gray", linewidth=1)
plt.axvline(250, color="red", linestyle=":", linewidth=1)
plt.xscale("symlog", linthresh=25)
plt.xlabel("Recovery step")
plt.ylabel("Per-seed delta_G (gaussian - GM)")
plt.title("Seed-level delta_G(t) trajectories vs recovery steps")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/conv_plot3_seed_trajectories.png", dpi=150)
plt.close()

# 4. fraction of images improved (win rate) vs steps
plt.figure(figsize=(7, 5))
plt.plot(steps, [r[str(s)]["win_rate_g"] for s in steps], "o-", color="C0", label="GM beats Gaussian (per-image)")
plt.plot(steps, [r[str(s)]["win_rate_m"] for s in steps], "s-", color="C1", label="GM beats Mask (per-image)")
plt.axhline(0.5, color="gray", linewidth=1)
plt.axvline(250, color="red", linestyle=":", linewidth=1)
plt.xscale("symlog", linthresh=25)
plt.xlabel("Recovery step")
plt.ylabel("Fraction of images where GM wins")
plt.title("Per-image GM win rate vs recovery steps")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{SP}/conv_plot4_win_rate_vs_steps.png", dpi=150)
plt.close()

# 5. MSE vs steps
plt.figure(figsize=(7, 5))
plt.plot(steps, [r[str(s)]["mean_mse_gaussian"] for s in steps], "o-", label="Gaussian-only")
plt.plot(steps, [r[str(s)]["mean_mse_mask"] for s in steps], "s-", label="Mask-only")
plt.plot(steps, [r[str(s)]["mean_mse_gm"] for s in steps], "^-", label="Gaussian+mask 1:1")
plt.axvline(250, color="red", linestyle=":", linewidth=1, label="original 250-step horizon")
plt.xscale("symlog", linthresh=25)
plt.xlabel("Recovery step")
plt.ylabel("Mean MSE")
plt.title("MSE vs recovery steps -- rises past step 250 despite LPIPS\nstill nominally improving (overshoot / pixel-level drift)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/conv_plot5_mse_vs_steps.png", dpi=150)
plt.close()

print("wrote 5 convergence plots")
