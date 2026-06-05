#!/usr/bin/env bash
# Safe monitoring / quota guard for the v13 SE2 bridge run (+ IN-1K resumes).
# Runs LOCALLY; SSHes via scripts/cluster/ssh.sh. READ-ONLY (no deletion).
# Reports: job state/elapsed, latest epoch, FID if done, home usage, ckpt dir sizes,
# operator_diag, and a log tail for any FAILED job.
#
# Usage:  bash projects/diff-EqM/experiments/bridge_poll.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SSH="$ROOT/scripts/cluster/ssh.sh"
RR=/n/home03/mkrasnow/research-repo
LOGS=$RR/projects/diff-EqM/slurm/logs
RES=$RR/projects/diff-EqM/results

BRIDGE="19404840 19404844 19404851 19405301"          # vK_tc, v13_disc, v13_rand, v10
IN1K="19399129 19399148 19399165 19399426"            # gate s1/s2, van-s2, lambda10

bash "$SSH" "
echo '=== BRIDGE v13 arms ==='
squeue -j ${BRIDGE// /,} -o '%.10i %.10P %.9T %.10M %.16R' 2>/dev/null
echo '=== IN-1K resumes ==='
squeue -j ${IN1K// /,} -o '%.10i %.10P %.9T %.10M %.16R' 2>/dev/null
echo '=== HOME quota ==='; df -h /n/home03/mkrasnow | tail -1
echo '=== HOLYLABS (IN-1K ckpts) ==='; df -h /n/holylabs 2>/dev/null | tail -1
now=\$(date +%s)
echo '=== per bridge arm ==='
for j in $BRIDGE; do
  f=\$(ls -t $LOGS/variant-pilot_\${j}_*.out 2>/dev/null | head -1)
  st=\$(sacct -j \$j --format=State -n 2>/dev/null | head -1 | tr -d ' ')
  if [ -n \"\$f\" ]; then
    age=\$(( (now-\$(stat -c %Y \"\$f\"))/60 ))
    ep=\$(grep -iE 'epoch ' \"\$f\" 2>/dev/null | tail -1)
    fid=\$(grep -iE 'cifar10_variant_fid' \"\$f\" 2>/dev/null | tail -1)
    op=\$(grep -iE 'operator|\\[v13\\]|\\[vK' \"\$f\" 2>/dev/null | tail -1)
    echo \"[\$j st=\$st age=\${age}min] \${ep:-no-epoch}\"
    [ -n \"\$op\" ] && echo \"     \$op\"
    [ -n \"\$fid\" ] && echo \"     \$fid\"
    case \"\$st\" in *FAIL*|*CANCEL*|*TIMEOUT*) echo '     --- ERR TAIL ---'; tail -8 \"\${f%.out}.err\" 2>/dev/null | sed 's/^/     /';; esac
  else echo \"[\$j st=\$st] no log yet\"; fi
done
echo '=== bridge ckpt dir sizes ==='; du -sh $RES/variant_v1*_19404* $RES/variant_v10_hard_example_19405* 2>/dev/null | sort -rh | head
echo '=== aggressive pruner alive? ==='; squeue -u \$USER -h -o '%j' 2>/dev/null | grep -c prune-aggr
"
