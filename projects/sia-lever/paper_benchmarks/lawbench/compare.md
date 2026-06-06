# LawBench comparison (fill after runs)

Record EXACTLY what was run. Do not compare to paper headline unless split/labels/budget match.

| System | Model | Split (n test) | Top-1 Acc | Notes |
|---|---|---|---|---|
| paper: initial | (paper) | full (913) | 13.5% | reference |
| paper: SIA-H | (paper) | full (913) | 50.0% | reference |
| paper: SIA-W+H | (paper) | full (913) | 70.1% | reference |
| ours: base gpt-oss-120b | openai/gpt-oss-120b | — | — | no training |
| ours: SIA-H (official loop) | gpt-oss-120b | — | — | harness loop only (public SIA) |
| ours: LoRA-SFT (W) | gpt-oss-120b + LoRA r32 | — | — | our W lane on train.csv |

Caveats to state in any writeup:
- Different target model than the paper (gpt-oss-120b vs paper's).
- If `--limit` was used anywhere, mark the row **reduced LawBench** and exclude from paper comparison.
- Public SIA performed only H; the W row is our addition (paper-style), not an exact reproduction.
