# Next Steps: Start Cluster Training

**Current Status:** SSH infrastructure ready, session not active
**Next Action:** Establish SSH connection, then submit training jobs

## ⚠️ SSH Connection Required

**Current SSH Status:**
```
✗ NO ACTIVE SSH SESSION
  Host: login.rc.fas.harvard.edu
  User: mkrasnow
  Reason: Never bootstrapped or session expired
```

## 🚀 Complete Workflow (30 minutes total)

### Step 1: Establish SSH Connection (5 minutes)

**Run this command:**
```bash
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh
```

**Expected output:**
```
Bootstrapping SSH ControlMaster session for mkrasnow@login.rc.fas.harvard.edu ...
Complete login + 2FA interactively when prompted.
```

**What you need to do:**
1. When prompted: Enter Harvard login password
2. When prompted: Enter 2FA code (phone/email)
3. Wait for: "SSH session ready."

**This establishes a persistent SSH connection for 30 minutes.**

---

### Step 2: Verify SSH is Active (1 minute)

**Run this command:**
```bash
scripts/cluster/ensure_session.sh
echo $?  # Should output: 0
```

**Expected output:**
```
(no output means it worked, exit code 0)
```

**If you see error:**
```
Exit code 21: No active SSH control session
```

Then re-run Step 1.

---

### Step 3: Submit All 5 Training Jobs (2 minutes)

**Run this command:**
```bash
cd projects/algebra-ebm
bash submit_cluster_training.sh
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════╗
║  Submitting Algebra EBM Training Jobs to Cluster          ║
╚════════════════════════════════════════════════════════════╝

Repository: /Users/mkrasnow/Desktop/research-repo
Project: algebra-ebm
Timestamp: 2026-02-11 12:34:56
Current git SHA: abc1234def5678...

Submitting training jobs...
────────────────────────────────────────────────────────────
Submitting: train_distribute.sbatch
  ✓ Submitted with Job ID: 12345678

Submitting: train_combine.sbatch
  ✓ Submitted with Job ID: 12345679

Submitting: train_isolate.sbatch
  ✓ Submitted with Job ID: 12345680

Submitting: train_divide.sbatch
  ✓ Submitted with Job ID: 12345681

Submitting: train_monolithic.sbatch (gpu partition)
  ✓ Submitted with Job ID: 12345682

════════════════════════════════════════════════════════════
All training jobs submitted successfully!
════════════════════════════════════════════════════════════

Job Summary:
  Distribute:    Job ID 12345678
  Combine:       Job ID 12345679
  Isolate:       Job ID 12345680
  Divide:        Job ID 12345681
  Monolithic:    Job ID 12345682

Tracking file saved: projects/algebra-ebm/.state/cluster_training.json
```

**This submits all 5 models to train in parallel.**

---

### Step 4: Monitor Training Progress (Check periodically over ~12 hours)

**Check job status:**
```bash
squeue -u $USER
```

**Expected output (while running):**
```
        JOBID     NAME     USER ST  START_TIME  NODES CPUS GRES
    12345678  algebra_t  mkrasnow  R 14:35:00      1    4 gpu:1
    12345679  algebra_t  mkrasnow  R 14:35:15      1    4 gpu:1
    12345680  algebra_t  mkrasnow  R 14:35:30      1    4 gpu:1
    12345681  algebra_t  mkrasnow  R 14:35:45      1    4 gpu:1
    12345682  algebra_t  mkrasnow  R 14:36:00      1    4 gpu:1
```

**Check detailed job info:**
```bash
sacct -j 12345678 --format=JobID,JobName,State,ExitCode,Elapsed
```

**View logs in real-time:**
```bash
ssh cluster.local
tail -f /n/home03/mkrasnow/research-repo/projects/algebra-ebm/slurm/logs/algebra_train_distribute_*.out
```

---

### Step 5: Verify Models Trained (After ~12 hours)

**Check models on cluster:**
```bash
ssh cluster.local "ls -lh /n/home03/mkrasnow/research-repo/projects/algebra-ebm/results/"
```

**Expected output:**
```
distribute/model.pt      1.2G  Feb 11 15:45
combine/model.pt         1.2G  Feb 11 16:02
isolate/model.pt         1.2G  Feb 11 16:19
divide/model.pt          1.2G  Feb 11 16:36
monolithic/model.pt      1.5G  Feb 11 17:15
```

---

### Step 6: Run Evaluation Experiments (2-3 hours)

**After all training jobs complete, run evaluations:**
```bash
cd projects/algebra-ebm
python run_experiments.py
```

**This will run 6 comprehensive evaluation experiments:**
1. Single-rule baseline (distribute, combine, isolate, divide)
2. Multi-rule 2-rules
3. Multi-rule 3-rules
4. Multi-rule 4-rules
5. Constrained inference
6. Compositional vs monolithic comparison

**Results saved to:**
```
projects/algebra-ebm/runs/
├── exp_001_single_rule_baseline_20260211_123456/results/
├── exp_002_multi_rule_2_20260211_*.../results/
├── exp_003_multi_rule_3_20260211_*.../results/
├── exp_004_multi_rule_4_20260211_*.../results/
├── exp_005_constrained_20260211_*.../results/
└── exp_007_comparison_20260211_*.../results/
```

---

## ⏱️ Timeline

| Step | Task | Time | Notes |
|------|------|------|-------|
| 1 | Bootstrap SSH | 5 min | Interactive (2FA required) |
| 2 | Verify SSH | 1 min | Should return exit code 0 |
| 3 | Submit jobs | 2 min | 5 jobs submitted, parallel |
| 4 | Wait for training | ~12 hours | Monitor with squeue |
| 5 | Verify models | 1 min | Check cluster storage |
| 6 | Run evaluation | 2-3 hours | 6 experiments |
| **Total** | **Complete workflow** | **~15 hours** | (most is waiting) |

---

## 📋 Checklist

Before you start:

- [ ] Read this document
- [ ] Check SSH key exists: `ls -la ~/.ssh/id_ed25519`
- [ ] Check cluster config: `cat .claude/cluster.local.json`
- [ ] Review SBATCH files: `ls -la projects/algebra-ebm/slurm/train_*.sbatch`
- [ ] Review documentation: `cat projects/algebra-ebm/CLUSTER_TRAINING_GUIDE.md`

Ready to start:

- [ ] Run: `scripts/cluster/ssh_bootstrap.sh` (enter password + 2FA)
- [ ] Verify: `scripts/cluster/ensure_session.sh`
- [ ] Submit: `cd projects/algebra-ebm && bash submit_cluster_training.sh`
- [ ] Monitor: `squeue -u $USER`
- [ ] Wait: ~12 hours for training to complete
- [ ] Evaluate: `python run_experiments.py`
- [ ] Analyze: Check `runs/*/results/` for evaluation results

---

## 🔧 Troubleshooting

### SSH Won't Connect

**Error:** "Permission denied"

**Solution:**
```bash
# Check SSH keys
ls -la ~/.ssh/id_ed25519

# Try manual SSH to debug
ssh -v mkrasnow@login.rc.fas.harvard.edu

# Run bootstrap again
scripts/cluster/ssh_bootstrap.sh
```

### Job Submission Failed

**Error:** "sbatch: error:"

**Solution:**
```bash
# Verify SSH is still active
scripts/cluster/ensure_session.sh

# If not, re-bootstrap
scripts/cluster/ssh_bootstrap.sh

# Try submission again
cd projects/algebra-ebm
bash submit_cluster_training.sh
```

### SSH Session Expired

**Error:** "Exit code 21: No active SSH control session"

**Solution:**
```bash
# Simply re-run bootstrap (no additional login needed if within 30 min)
scripts/cluster/ssh_bootstrap.sh
```

---

## 📚 Documentation

For more information:

- **SSH Setup Details:** `SSH_SETUP_REQUIRED.md`
- **Cluster Infrastructure:** `CLUSTER_INFRASTRUCTURE_ANALYSIS.md`
- **Training Guide:** `projects/algebra-ebm/CLUSTER_TRAINING_GUIDE.md`
- **Experiment Design:** `projects/algebra-ebm/documentation/experiment-plan.md`

---

## 🎯 Key Points

1. **SSH First:** Must establish SSH connection before submitting jobs
2. **One-time Setup:** SSH bootstrap only needed if session expires
3. **Parallel Training:** All 5 models train simultaneously (~12 hours total)
4. **Automatic Storage:** Models auto-synced to cluster persistent storage
5. **Complete Evaluation:** 6 comprehensive experiments test all scenarios

---

## NOW: Ready to Begin?

**Execute these 3 commands:**

```bash
# 1. Establish SSH (interactive, 2FA required)
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh

# 2. Submit training jobs (no interaction needed)
cd projects/algebra-ebm
bash submit_cluster_training.sh

# 3. Monitor progress (check periodically)
squeue -u $USER
```

**That's it! Models will train on cluster GPUs and be ready for evaluation in ~12 hours.**

---

**Let me know when SSH is ready to establish!**

