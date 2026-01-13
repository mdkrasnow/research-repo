# SLURM (Harvard FAS RC + general)

Harvard FAS RC documentation:
https://docs.rc.fas.harvard.edu/kb/category/cluster-usage/

## Remote SLURM from a laptop (recommended)
If you're running Claude Code on macOS/Windows, `sbatch` will not exist locally.
This repo supports a **hybrid policy**:

1) Establish an SSH session to the FASRC login node once (includes 2FA)
2) Reuse that session via SSH ControlMaster for ~30 minutes
3) Submit + poll jobs remotely, then sync logs back locally

### Setup
1. Copy config:
   - `.claude/cluster.example.json` → `.claude/cluster.local.json`
2. Fill in:
   - `ssh.user`
   - `remote.repo_root` (path where this repo exists on the cluster)
3. Bootstrap SSH session (interactive, includes 2FA):
   - `scripts/cluster/ssh_bootstrap.sh`

After that, `/dispatch` and `/run-experiments` can submit to SLURM remotely.

### Security notes
- Never store passwords or 2FA codes in repo files.
- Prefer SSH keys + agent forwarding where possible.

## Core commands
- `sinfo`
- `squeue -u $USER`
- `sacct -j <jobid> --format=...`
- `sbatch <script.sbatch>`
- `scancel <jobid>`

## Sbatch skeleton
```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=projects/<slug>/slurm/logs/%x_%j.out
#SBATCH --error=projects/<slug>/slurm/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Optional:
# SBATCH --gres=gpu:1
# SBATCH --partition=<partition>
# SBATCH --constraint=<constraint>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python -m experiments.run --config configs/exp.yaml
```

## Repo contract
- Generated scripts live in `projects/<slug>/slurm/jobs/`
- Logs go to `projects/<slug>/slurm/logs/`
- Every submission gets a run ledger under `projects/<slug>/runs/<run_id>/`

## Helper scripts
- Local/remote submit: `scripts/cluster/submit.sh <sbatch> <project_slug>`
- Local/remote status: `scripts/cluster/status.sh <jobid>`
- Remote log sync: `scripts/cluster/remote_fetch.sh <project_slug>`
