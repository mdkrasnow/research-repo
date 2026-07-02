# Literature Notes — diff-EqM

Per-paper structured notes feeding our positioning, mechanism design, and risk assessment. Lives at `projects/diff-EqM/documentation/literature/`.

## Why
Position the paper. Only work on things meaningful to the field. Surface scoop risk early. Avoid literature laundering (CLAUDE.md rule).

## Note template (one file per paper)

Filename: `<lastname><year>_<short-slug>.md` — e.g. `wu2025_dat.md`, `madry2017_pgd.md`.

```markdown
# <Title>

**Citation**: Authors. *Title*. Venue Year. arxiv:XXXX.XXXXX. URL.
**Read date**: YYYY-MM-DD
**Read by**: claude | mk
**Read depth**: skim | abstract+method | full | full+code

## Setting
- Task: <generative modeling | discriminative | hybrid>
- Model family: <classical EBM | score | flow | diffusion | EqM | ...>
- Dataset(s): <CIFAR-10 | IN-256 | ...>
- Best FID / metric: <number, conditions>

## Method (in one paragraph + equations if salient)
<...>

## Key technical choices
- Loss: <exact formulation>
- Training: <optimizer, schedule, batch size if non-standard>
- Sampler / inference: <details>
- Adversarial / mining (if any): <PGD steps, eps, objective>

## What this paper supports
<Concrete claims the paper makes that we can cite. NOT general vibes.>

## What this paper does NOT support
<Adjacent claims people might misattribute. CLAUDE.md rule: citation ≠ mechanism.>

## Relevance to v10 / diff-EqM
**Threat level**: HIGH (scoop / direct competitor) | MEDIUM (overlapping ideas) | LOW (background) | INFO (cite for completeness)
**Differentiation from us**: <one sentence>
**Action**: <cite in related work | inform v10 hyperparameter X | inform framing Y | watch for follow-ups | no action>

## Open questions / takeaways for our work
<Anything that should change our plan, code, or framing.>

## Reference links
- arxiv: ...
- code (if any): ...
- project page: ...
- citing this paper (recent): <list relevant follow-ups we should also read>
```

## Synthesis output
After reading the target slate, write `documentation/literature/SYNTHESIS.md` summarizing:
1. Updated positioning (v10's contribution restated against the read landscape).
2. Mechanism-design adjustments (HP changes, alternative formulations to consider before launch).
3. Risk register updates (new scoop risks; new threats).
4. Recommended CLAUDE.md / summer-plan / phase-0-spec edits.

## Read slate — Phase 0
See `documentation/phase-0-spec.md` Task 0.A.1 for the locked list.
