# LawBench lane (paper-territory stretch)

LawBench (criminal-charge prediction) is **bundled in the public SIA repo**
(`baselines/vendor/sia/sia/tasks/lawbench`): 913 test cases, 191 charge labels, a labeled
`train.csv` (good for a real W-update), and an `evaluate.py`.

Paper reference numbers (top-1 accuracy):

| System | Accuracy |
|---|---|
| initial | 13.5% |
| SIA-H | 50.0% |
| SIA-W+H | 70.1% |

**Caveat (read before comparing):** only compare to these if you use the **same data split, label
set, and task definition** with a comparable model/budget. The paper used its own target model; we
use gpt-oss-120b. If you run a reduced split (time/GPU), label it **reduced LawBench** and do NOT
compare to the paper headline.

## Lanes
1. **Official SIA-H** (harness loop, no weights) on LawBench with a gpt-oss-120b target:
   ```bash
   bash baselines/official_sia/run_lawbench_sia_h.sh
   python3 baselines/official_sia/collect_runs.py --run-id <id>
   ```
2. **W-update (our addition; public SIA has no W code):** LoRA-SFT gpt-oss-120b on the labeled
   `train.csv` (text → charge), then evaluate top-1 on the held-out test:
   ```bash
   python3 paper_benchmarks/lawbench/train_lora_sft.py --base-model openai/gpt-oss-120b \
       --out adapters/gpt_oss_120b/lawbench_sft_<ts>
   python3 paper_benchmarks/lawbench/eval_lora.py --adapter adapters/gpt_oss_120b/lawbench_sft_<ts>
   ```
3. **Export SIA-H rollouts** (for distillation / inspection):
   ```bash
   python3 paper_benchmarks/lawbench/export_rollouts.py --run-id <id>
   ```

## Data locations (vendored)
- train: `baselines/vendor/sia/sia/tasks/lawbench/data/training_data/train.csv`
- test (no labels): `.../data/public/test.csv`  • labels: `.../data/private/test.csv`
- labels list: `.../data/public/classes.json`
- official evaluator: `.../data/public/evaluate.py`

## Honesty
- Public SIA = **harness loop only**. Lane 2 (LoRA W-update) is **our** implementation, labeled as a
  paper-STYLE W update, not an exact paper reproduction.
- Record model id, split size, LoRA config, and GPU budget in `compare.md` for every run.

## Do-not-block rule
LawBench must not block the SIA-Lever-120B demo. Run it only after SIA-Lever-120B (Lanes A/B) work.
