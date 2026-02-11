# Cluster Infrastructure Analysis

**Date:** 2026-02-11
**Status:** SSH Setup Required Before Training

## Executive Summary

The codebase has a well-architected cluster infrastructure for submitting and monitoring SLURM jobs. **SSH must be established first** before submitting training jobs.

### Current Status
```
✓ SSH infrastructure configured
✓ Cluster configuration present
✗ SSH session NOT active (requires manual bootstrap)
```

## Infrastructure Components

### 1. Configuration Files

**`.claude/cluster.local.json`** - Central cluster configuration
```json
{
  "ssh": {
    "user": "mkrasnow",
    "host": "login.rc.fas.harvard.edu",
    "port": 22,
    "control_path": "~/.ssh/cc-research-repo-%r@%h:%p",
    "control_persist": "30m"
  },
  "remote": {
    "repo_root": "/n/home03/mkrasnow/research-repo",
    "rsync_excludes": [...]
  },
  "git": {
    "url": "https://github.com/mdkrasnow/research-repo.git",
    "default_branch": "main"
  }
}
```

**Key Details:**
- **Host:** `login.rc.fas.harvard.edu` (Harvard RC cluster)
- **User:** `mkrasnow` (cluster username)
- **Control Path:** Persistent SSH socket in `~/.ssh/`
- **Control Persist:** 30 minutes of inactivity tolerance
- **Remote Root:** `/n/home03/mkrasnow/research-repo` (persistent storage)

### 2. SSH Infrastructure Scripts

**`scripts/cluster/ssh_bootstrap.sh`** (72 lines)
```
Purpose: Establish initial SSH ControlMaster session
Flow:
  1. Read cluster config from .claude/cluster.local.json
  2. Extract SSH parameters (user, host, port)
  3. Create SSH control socket
  4. Prompt for interactive login + 2FA
  5. Keep session alive for 30 minutes
Requires: Interactive terminal (password + 2FA)
Output: "SSH session ready."
```

**`scripts/cluster/ensure_session.sh`** (52 lines)
```
Purpose: Check if SSH session is active
Flow:
  1. Read cluster config
  2. Check for existing control socket
  3. Return 0 if active, 21 if not
Output: Exit code 0 (active) or 21 (inactive)
```

**`scripts/cluster/ssh.sh`** (44 lines)
```
Purpose: Execute commands over SSH using control socket
Flow:
  1. Use existing ControlMaster session
  2. Execute remote command in BatchMode
  3. Don't prompt for passwords (uses existing socket)
Usage: ssh.sh "remote_command"
Requires: Active SSH session (from ssh_bootstrap.sh)
```

**`scripts/cluster/submit.sh`** (57 lines)
```
Purpose: Submit SLURM jobs to cluster
Flow:
  1. Check if sbatch is available locally
  2. If not, use remote_submit.sh (SSH-based submission)
  3. Extract and return job ID
Usage: submit.sh <sbatch_file> <project_slug>
```

**`scripts/cluster/remote_submit.sh`** (76 lines)
```
Purpose: Remote SLURM submission via SSH
Flow:
  1. Sync local slurm/ directory to cluster
  2. Extract SSH parameters from config
  3. Submit sbatch job remotely with git vars
  4. Return job ID
Uses: SSH control socket for efficiency
Key Feature: Auto-syncs code and git SHA
```

### 3. Cluster Information

**Cluster: Harvard RC (Research Computing)**
- **Login Node:** login.rc.fas.harvard.edu
- **Job Scheduler:** SLURM
- **GPU Available:** A100 GPUs
- **Storage:** /n/home03/ (persistent home directory)
- **Temp:** /tmp (local to compute nodes)

**Partitions Available:**
- `gpu_test`: Higher priority, 24-hour limit (good for quick jobs)
- `gpu`: Lower priority, longer time limits (for longer jobs)

## Workflow: How It Works

### Phase 1: SSH Bootstrap (One-time, Interactive)

```
User runs: scripts/cluster/ssh_bootstrap.sh

ssh_bootstrap.sh:
  ├─ Read .claude/cluster.local.json
  ├─ Extract: user=mkrasnow, host=login.rc.fas.harvard.edu
  ├─ Create SSH control socket at ~/.ssh/cc-research-repo-*
  └─ Prompt: "Complete login + 2FA interactively"
      ├─ User enters: password
      ├─ User enters: 2FA code
      └─ SSH session established, kept alive 30 minutes

Result: SSH ControlMaster socket created
        All subsequent commands use this socket (no 2FA needed)
```

### Phase 2: Job Submission (Uses SSH Socket)

```
User runs: bash submit_cluster_training.sh

submit_cluster_training.sh:
  ├─ Loop through 5 training models
  ├─ For each model:
  │   └─ Call: scripts/cluster/submit.sh <sbatch_file> <project>
  │       ├─ Check if sbatch exists locally (no, it doesn't)
  │       ├─ Call: scripts/cluster/remote_submit.sh
  │           ├─ Extract SSH params from .claude/cluster.local.json
  │           ├─ Sync local slurm/ → remote slurm/ via SSH
  │           ├─ Call: scripts/cluster/ssh.sh "cd /n/home03/... && sbatch ..."
  │           │   └─ Uses existing SSH socket (no 2FA!)
  │           └─ Return job ID (e.g., 12345)
  │       └─ Return job ID
  └─ Display all job IDs

Result: 5 SLURM jobs submitted to cluster
        Job IDs displayed and saved to cluster_training.json
```

### Phase 3: Monitoring (Uses SSH Socket)

```
User runs: squeue -u $USER

squeue:
  ├─ System detects SSH socket exists
  └─ Executes remotely on cluster (via socket)

Result: Job status displayed
```

## Key Files and Flow

```
Local Machine (~/)
├── .ssh/
│   ├── id_ed25519                          (Private SSH key)
│   ├── id_ed25519.pub                      (Public SSH key)
│   └── cc-research-repo-*                  (ControlMaster socket)
│
└── Desktop/research-repo/
    ├── .claude/
    │   └── cluster.local.json              (Config: user, host, remote path)
    │
    ├── scripts/cluster/
    │   ├── ssh_bootstrap.sh                (Establish SSH)
    │   ├── ensure_session.sh               (Check SSH)
    │   ├── ssh.sh                          (Execute remote commands)
    │   ├── submit.sh                       (Submit jobs locally or remote)
    │   ├── remote_submit.sh                (Remote submission)
    │   └── ...
    │
    └── projects/algebra-ebm/
        ├── submit_cluster_training.sh      (Master orchestrator)
        ├── slurm/
        │   ├── train_*.sbatch              (SLURM job scripts)
        │   └── logs/                       (Job output files)
        └── ...

Cluster (Harvard RC)
├── login.rc.fas.harvard.edu
│   └── /n/home03/mkrasnow/research-repo/  (Persistent storage)
│       ├── projects/algebra-ebm/
        │   ├── slurm/                      (Synced SBATCH files)
        │   └── results/                    (Trained models)
│       └── ...
│
└── GPU Compute Nodes
    └── /tmp/algebra-train-<JOB_ID>/        (Temporary work directory)
        ├── Repository cloned               (Fresh clone per job)
        ├── Models trained here             (GPU acceleration)
        ├── Results synced back             (Auto-copy to persistent)
        └── Clean up on completion
```

## SSH Security & Reliability

### ControlMaster Benefits

1. **Reduces Auth Overhead**
   - One login, many commands
   - No repeated 2FA prompts

2. **Multiplexing**
   - Multiple concurrent SSH channels
   - Faster command execution

3. **Persistence**
   - Default: Keeps socket alive 30 minutes
   - Automatic cleanup on timeout

4. **Batching**
   - Commands use BatchMode=yes
   - No interactive prompts in scripts

### SSH Socket Management

```bash
# Check active sockets
ls -la ~/.ssh/cc-research-repo-*

# Verify connection
scripts/cluster/ensure_session.sh

# Manually close session
ssh -O exit mkrasnow@login.rc.fas.harvard.edu
```

## Data Flow: Training Submission

```
Step 1: Bootstrap SSH
  └─ scripts/cluster/ssh_bootstrap.sh
     ├─ User interactive login + 2FA
     ├─ SSH ControlMaster socket created
     └─ Session valid for 30 minutes

Step 2: Submit Training
  └─ bash submit_cluster_training.sh
     ├─ Extracts git SHA from local repo
     ├─ For each of 5 models:
     │  └─ scripts/cluster/submit.sh <sbatch> <project>
     │     ├─ Syncs projects/algebra-ebm/slurm/ → cluster
     │     └─ sbatch --export=GIT_SHA=$sha <sbatch_file>
     │        └─ Job ID returned (e.g., 12345)
     └─ Create cluster_training.json with all job IDs

Step 3: SLURM Job Execution (on cluster)
  └─ SLURM runs: sbatch
     ├─ Create temp work dir: /tmp/algebra-train-12345/
     ├─ Clone repo: git clone $GIT_URL /tmp/algebra-train-12345/
     ├─ Checkout: git checkout $GIT_SHA
     ├─ Load modules: python/3.10, cuda/11.8
     ├─ Run training: python train_algebra.py --rule distribute
     ├─ Sync results: rsync /tmp/...results/ → /n/home03/.../results/
     ├─ Cleanup: rm -rf /tmp/algebra-train-12345/
     └─ Log output: projects/algebra-ebm/slurm/logs/

Step 4: Monitor (from local machine)
  └─ squeue -u $USER
     ├─ Uses SSH socket (already established)
     └─ Displays job status on cluster
```

## Failure Modes & Recovery

### SSH Disconnection (Exit Code 21)

**Symptom:**
```
Error: No active SSH control session for mkrasnow@...
Run: scripts/cluster/ssh_bootstrap.sh
```

**Cause:**
- Session timeout (30 minutes)
- Network interruption
- Manual termination

**Recovery:**
```bash
scripts/cluster/ssh_bootstrap.sh  # Re-establish
```

### Job Submission Failure

**Symptom:**
```
sbatch: error: Job submission failed
```

**Causes & Solutions:**
1. SSH not active: Run `ssh_bootstrap.sh`
2. File permission: Check slurm/ directory permissions
3. Cluster quota: Check disk space on /n/home03/
4. SLURM error: Check SBATCH file syntax

**Debugging:**
```bash
scripts/cluster/ssh.sh "sbatch --help"  # Verify sbatch available
ssh cluster.local                        # Manual login for investigation
```

### Git SHA Mismatch

**Symptom:**
```
Repository cloned but not at correct SHA
```

**Cause:**
- Git branch doesn't contain commit
- Remote branch not up-to-date

**Solution:**
```bash
# Ensure commit is pushed
git push origin main

# Then resubmit
bash submit_cluster_training.sh
```

## Configuration Customization

### Changing SSH Timeout

Edit `.claude/cluster.local.json`:
```json
"control_persist": "2h"  // Change from "30m" to "2h"
```

Then bootstrap again:
```bash
scripts/cluster/ssh_bootstrap.sh
```

### Changing Cluster Host

Edit `.claude/cluster.local.json`:
```json
"host": "different.cluster.com"
"user": "different_user"
```

### Changing Remote Storage Path

Edit `.claude/cluster.local.json`:
```json
"remote": {
  "repo_root": "/different/path/research-repo"
}
```

## Summary: Complete Setup

### Before Any Training

1. **Cluster config exists** ✓ (`.claude/cluster.local.json`)
2. **SSH keys exist** ✓ (`~/.ssh/id_ed25519`)
3. **Scripts are in place** ✓ (`scripts/cluster/`)
4. **SBATCH files ready** ✓ (`projects/algebra-ebm/slurm/`)

### What You Need to Do

1. **Run once (interactive):** `scripts/cluster/ssh_bootstrap.sh`
   - Enter password
   - Enter 2FA code
   - Session established

2. **Then submit jobs:** `bash submit_cluster_training.sh`
   - No additional interaction needed
   - Jobs submitted automatically
   - Job IDs displayed

3. **Monitor:** `squeue -u $USER`
   - Uses existing SSH session
   - No additional auth needed

## Conclusion

The infrastructure is **fully configured and ready**. The only requirement is:

**Establish SSH connection once:**
```bash
scripts/cluster/ssh_bootstrap.sh
```

After that, all training submission and monitoring is automated!

