# YES - SSH Connection Required Before Training

## Quick Answer

**Q: Do we need to establish SSH connection first?**

**A: YES. One SSH bootstrap command must be run before submitting training jobs.**

```bash
scripts/cluster/ssh_bootstrap.sh  # Run once, interactive (5 minutes)
```

After this, everything else is automated.

---

## Current SSH Status

```
Status: NO ACTIVE SSH SESSION
Host: login.rc.fas.harvard.edu (Harvard RC cluster)
User: mkrasnow

What's missing: SSH ControlMaster socket
What you need: One-time interactive login
```

## What Happens When You Run SSH Bootstrap

```bash
$ scripts/cluster/ssh_bootstrap.sh

Process:
1. Read cluster configuration from .claude/cluster.local.json
2. Prepare SSH ControlMaster socket location
3. Prompt: "Complete login + 2FA interactively when prompted."
4. You enter: Harvard password
5. You enter: 2FA code (phone/email/authenticator)
6. SSH socket created: ~/.ssh/cc-research-repo-mkrasnow@login.rc.fas.harvard.edu:22
7. Socket valid for: 30 minutes (configurable)

Result: "SSH session ready."
        All subsequent commands use this socket (no 2FA needed)
```

## Why SSH is Required

The cluster submission uses SSH to:

1. **Sync code** - Copy SBATCH files to cluster
2. **Submit jobs** - Send sbatch command to cluster
3. **Monitor jobs** - Check SLURM status
4. **Manage files** - Check results on cluster

All of this goes through the SSH control socket.

## After SSH Bootstrap

Once SSH socket is active, everything is automatic:

```bash
# 1. SSH already established (from previous bootstrap)
# 2. Submit jobs (no additional auth)
$ cd projects/algebra-ebm && bash submit_cluster_training.sh

# 3. Monitor (uses existing socket)
$ squeue -u $USER

# 4. Check logs
$ ssh cluster.local  # Uses existing socket, no prompt
```

## SSH Infrastructure Explained

The codebase uses **SSH ControlMaster** for efficiency:

```
SSH ControlMaster:
  ├─ One-time login with 2FA
  ├─ Creates persistent socket
  ├─ Multiplexes multiple commands
  ├─ No repeated authentication
  └─ Configurable timeout (default 30 min)

Benefits:
  ✓ Only one 2FA code needed
  ✓ Subsequent commands are instant
  ✓ Multiple terminals can share socket
  ✓ Clean failure handling
```

## Complete 3-Step Workflow

### Step 1: SSH Bootstrap (5 min, interactive)
```bash
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh
# Enter password + 2FA code
```

### Step 2: Submit Training (2 min, automatic)
```bash
cd projects/algebra-ebm
bash submit_cluster_training.sh
# 5 jobs submitted, no additional input
```

### Step 3: Monitor (ongoing, automatic)
```bash
squeue -u $USER
# Check progress, uses existing SSH socket
```

## SSH Verification

Before submitting, verify SSH is active:

```bash
scripts/cluster/ensure_session.sh
echo $?
```

**Expected output:** Exit code 0 (active)

**If you get exit code 21:** Session expired or not established
- Just re-run: `scripts/cluster/ssh_bootstrap.sh`

## What Happens Behind the Scenes

When SSH is bootstrap'd:

```
Local ~/.ssh/
└─ cc-research-repo-mkrasnow@login.rc.fas.harvard.edu:22
   (SSH ControlMaster socket created)
        ↓
   Connects to: login.rc.fas.harvard.edu
        ↓
   Provides access to: /n/home03/mkrasnow/research-repo/
        ↓
   Used by: All cluster commands
        ↓
   Multiplexes: Job submission, monitoring, file sync
```

## SSH Session Lifecycle

```
Timeline:

T=0: Run ssh_bootstrap.sh
T=2min: Enter password
T=3min: Enter 2FA code
T=5min: "SSH session ready."
        Socket created, valid until T=35min

T=5min-30min: SSH socket available
  - Can submit jobs
  - Can monitor
  - Can manage files
  - No additional auth needed

T=30min+: SSH socket expires
  - Need to re-run bootstrap
  - Just run ssh_bootstrap.sh again
  - Takes 5 minutes
```

## Failure Handling

### If SSH Session Dies

```bash
Error: Exit code 21: No active SSH control session

Solution: Re-establish
$ scripts/cluster/ssh_bootstrap.sh
```

### If You See "Permission denied"

```bash
Error: Permission denied (publickey,password,keyboard-interactive)

Solution:
1. Check SSH keys: ls -la ~/.ssh/id_ed25519
2. Verify config: cat .claude/cluster.local.json
3. Try bootstrap: scripts/cluster/ssh_bootstrap.sh
```

### If 2FA Prompt Doesn't Appear

```bash
Issue: Running in non-interactive terminal

Solution:
$ scripts/cluster/ssh_bootstrap.sh  # Run in interactive terminal
# NOT in background: scripts/cluster/ssh_bootstrap.sh &
```

## Summary

| When | What | Required | Duration |
|------|------|----------|----------|
| First time | SSH bootstrap | YES, interactive | 5 min |
| Job submit | submit_cluster_training.sh | NO (uses socket) | 2 min |
| Monitoring | squeue | NO (uses socket) | Ongoing |
| Session expires | SSH bootstrap again | YES, interactive | 5 min |

## Next Actions

1. ✅ Understand: SSH required (this document)
2. ✅ Examine: Cluster infrastructure (explored above)
3. ▶️ **Execute: SSH bootstrap** (next step)

```bash
cd /Users/mkrasnow/Desktop/research-repo
scripts/cluster/ssh_bootstrap.sh
# Enter password + 2FA
```

4. ▶️ **Submit: Training jobs** (after SSH ready)

```bash
cd projects/algebra-ebm
bash submit_cluster_training.sh
```

5. ✅ Monitor: Job progress

```bash
squeue -u $USER
```

---

## Documentation Reference

- **This file:** SSH_REQUIRED_ANSWER.md (Quick answer)
- **SSH Setup Details:** SSH_SETUP_REQUIRED.md (Complete guide)
- **Infrastructure:** CLUSTER_INFRASTRUCTURE_ANALYSIS.md (Full analysis)
- **Workflow:** NEXT_STEPS_CLUSTER_TRAINING.md (Step-by-step)

---

**Bottom Line:**

Yes, SSH bootstrap is required, but it's a one-time 5-minute interactive step. After that, everything is automatic. The infrastructure is fully configured and ready to go.

