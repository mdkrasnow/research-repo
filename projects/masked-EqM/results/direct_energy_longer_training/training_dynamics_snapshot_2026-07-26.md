# Live training-dynamics snapshot

This snapshot uses the continuation `log.txt` files directly; no GPU job is
needed. Slopes are ordinary least-squares fits of logged loss against optimizer
step. The 500-point window is roughly the last 25,000 optimizer steps.

| arm | step | loss mean (last 500) | slope / step (last 500) | loss SD | steps/sec |
|---|---:|---:|---:|---:|---:|
| none | 182,800 | 10.7450 | -1.74e-6 | 0.1080 | 8.89 |
| dot | 108,800 | 11.2535 | -1.47e-6 | 0.1062 | 4.75 |
| direct | 106,950 | 11.3026 | -1.90e-6 | 0.1009 | 4.75 |

The last 100 logged points are noisy and have slightly positive slopes for all
three arms (none +3.21e-6, dot +4.62e-6, direct +2.64e-6). That is minibatch
noise, not evidence of divergence: the wider 500-point fits remain negative
for every arm. Direct is currently about 0.05 loss above dot at comparable
step count, while none is lower but has progressed substantially farther.

## Epoch-1 field audit context

The completed fixed-batch audit on epoch-1 checkpoints gave mean field cosine
none 0.663, dot 0.662, direct 0.658; mean norm ratios were 0.638, 0.699,
and 0.665. Direct head/backbone gradient norms were larger (3.58/1.61) than
dot (2.69/1.18) and none (1.75/0.94), so the current loss gap is not explained
by absent direct gradients or directional collapse.

Epoch-2 field diagnostics are queued behind the active FID probes (`35379037`)
and will be written separately.
