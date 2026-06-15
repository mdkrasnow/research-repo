# CLUSTER DOWN — MGHPCC power downtime June 15–18, 2026

FASRC annual power downtime at the Holyoke (MGHPCC) data center. ALL login nodes,
storage, and compute (Cannon/FASSE/Kempner) are OFF.
- Mon Jun 15 9 AM: power-down begins
- Tue–Wed Jun 16–17: power out
- Thu Jun 18 ~5 PM: expected return to full service
- Fri Jun 19: university holiday

Port-22 timeouts to login.rc.fas.harvard.edu are the outage, NOT VPN/2FA.

## Queued (fire when cluster returns ~Jun 18)
Run `scripts/cluster/fire_overnight.sh` (idempotent) once SSH is reachable. It submits:
- maze GPU scale-up × 3 seeds (`maze_gpu.sbatch`)
- online 50k adaptive × 3 seeds (`online_seed.sbatch`)
Source: https://status.rc.fas.harvard.edu/
