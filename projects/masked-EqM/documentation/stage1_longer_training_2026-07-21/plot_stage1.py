import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(f"{SP}/stage1_analysis.json"))
epochs = d["epochs"]
r = d["results"]

# 1. delta_G vs epoch
plt.figure(figsize=(7, 5))
dg = [r[str(e)]["delta_g"] for e in epochs]
dg_lo = [r[str(e)]["ci_g"][0] for e in epochs]
dg_hi = [r[str(e)]["ci_g"][1] for e in epochs]
plt.plot(epochs, dg, "o-", color="C0")
plt.fill_between(epochs, dg_lo, dg_hi, color="C0", alpha=0.2)
plt.axhline(0, color="gray", linewidth=1)
plt.axhline(0.008, color="red", linestyle=":", linewidth=1.5, label="promotion threshold (0.008)")
plt.xlabel("Training epoch")
plt.ylabel("delta_G = LPIPS(gaussian) - LPIPS(GM)")
plt.title("delta_G vs training epoch (Stage 1, cutoff 0.10)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/stage1_plot1_delta_g_vs_epoch.png", dpi=150)
plt.close()

# 2. delta_M vs epoch
plt.figure(figsize=(7, 5))
dm = [r[str(e)]["delta_m"] for e in epochs]
dm_lo = [r[str(e)]["ci_m"][0] for e in epochs]
dm_hi = [r[str(e)]["ci_m"][1] for e in epochs]
plt.plot(epochs, dm, "s-", color="C1")
plt.fill_between(epochs, dm_lo, dm_hi, color="C1", alpha=0.2)
plt.axhline(0, color="gray", linewidth=1)
plt.xlabel("Training epoch")
plt.ylabel("delta_M = LPIPS(mask) - LPIPS(GM)")
plt.title("delta_M vs training epoch (Stage 1, cutoff 0.10)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/stage1_plot2_delta_m_vs_epoch.png", dpi=150)
plt.close()

# 3. absolute Fourier LPIPS vs epoch, all 3 families
plt.figure(figsize=(7, 5))
plt.plot(epochs, [r[str(e)]["mean_lpips_gaussian"] for e in epochs], "o-", label="Gaussian-only")
plt.plot(epochs, [r[str(e)]["mean_lpips_mask"] for e in epochs], "s-", label="Mask-only")
plt.plot(epochs, [r[str(e)]["mean_lpips_gm"] for e in epochs], "^-", label="Gaussian+mask 1:1")
plt.xlabel("Training epoch")
plt.ylabel("Mean Fourier-recovery LPIPS (cutoff 0.10)")
plt.title("Absolute Fourier-recovery LPIPS vs training epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/stage1_plot3_abs_lpips_vs_epoch.png", dpi=150)
plt.close()

# 4. seed-level delta_G trajectories
plt.figure(figsize=(7, 5))
for i, seed in enumerate(d["seeds"]):
    traj = [r[str(e)]["seed_lpips_gaussian"][i] - r[str(e)]["seed_lpips_gm"][i] for e in epochs]
    plt.plot(epochs, traj, "o-", alpha=0.8, label=f"seed{seed}")
plt.axhline(0, color="gray", linewidth=1)
plt.axhline(0.008, color="red", linestyle=":", linewidth=1, label="promotion threshold")
plt.xlabel("Training epoch")
plt.ylabel("Per-seed delta_G")
plt.title("Seed-level delta_G trajectories vs training epoch")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/stage1_plot4_seed_delta_g_trajectories.png", dpi=150)
plt.close()

# 5. FID vs epoch
plt.figure(figsize=(7, 5))
plt.plot(epochs, [r[str(e)]["mean_fid_gaussian"] for e in epochs], "o-", label="Gaussian-only")
plt.plot(epochs, [r[str(e)]["mean_fid_mask"] for e in epochs], "s-", label="Mask-only")
plt.plot(epochs, [r[str(e)]["mean_fid_gm"] for e in epochs], "^-", label="Gaussian+mask 1:1")
plt.xlabel("Training epoch")
plt.ylabel("FID (2000 samples, sanity-scale)")
plt.title("Generation FID vs training epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/stage1_plot5_fid_vs_epoch.png", dpi=150)
plt.close()

# 6. mask-recovery LPIPS vs epoch
plt.figure(figsize=(7, 5))
plt.plot(epochs, [r[str(e)]["mean_mr_lpips_gaussian"] for e in epochs], "o-", label="Gaussian-only")
plt.plot(epochs, [r[str(e)]["mean_mr_lpips_mask"] for e in epochs], "s-", label="Mask-only")
plt.plot(epochs, [r[str(e)]["mean_mr_lpips_gm"] for e in epochs], "^-", label="Gaussian+mask 1:1")
plt.xlabel("Training epoch")
plt.ylabel("Trained mask-recovery LPIPS (mask_prob=0.5)")
plt.title("Mask-recovery LPIPS vs training epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SP}/stage1_plot6_mask_recovery_vs_epoch.png", dpi=150)
plt.close()

print("wrote 6 Stage1 plots")
