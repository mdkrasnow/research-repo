# Live demo script — ~90 seconds

Spoken walkthrough to narrate while `scripts/demo.py` runs. Times are cues, not hard marks.

---

**[0:00 — set the stakes, before running]**

> "Here's a self-improving agent that's stopped improving. The question every one of these
> systems faces is: *what do I fix?* There are three levers — patch the evaluator, retrain
> the weights, or both. The default reflex is to retrain, because that's what 'self-improve'
> means to most loops. Retraining is also the expensive lever. Watch what happens when the
> bug isn't actually in the weights."

*(run `scripts/demo.py`)*

**[0:15 — the failure trace appears]**

> "This is the failure trace — and notice it's label-free. No one told the system what's
> broken. This is exactly what a deployed agent has: just the symptoms."

**[0:30 — the naive default fires]**

> "The naive policy does the default thing — it fires W, a full weight retrain. On this
> episode that's wasted: the evaluator was the problem, so the score won't move and we just
> paid for a fine-tune."

**[0:45 — SIA-Lever reads the same trace]**

> "Now SIA-Lever reads the *same* trace. It was trained by actually applying every lever and
> measuring what really fixed each failure — no hand-labels, no distilled targets. And it
> says: H — fix the evaluator first. Different call, from the same evidence."

**[1:00 — the cost delta — MONEY SHOT]**

> "Here's the whole pitch in one line:"
>
> **"The default says 'retrain the model.' SIA-Lever says 'your evaluator is broken — fix
> that first.' One of those costs a GPU fine-tune. One of them costs nothing."**

**[1:10 — scale it up]**

> "And this isn't just the toy. On a hard compound-fault benchmark our selector crushes
> regret 0.161 to 0.043 — without collapsing to a constant guess; we have a gate that caught
> our own false win. On a real AlphaFold-3 GPU kernel it reaches the right lever with 48
> retrains instead of a re-measuring scheduler's 72. A third fewer of the expensive calls."

**[1:25 — close]**

> "Self-improving agents are about to spend a lot of GPU. SIA-Lever tells them *which* fix to
> spend it on. Diagnose, then repair the right thing."

---

**If the demo errors mid-run:** fall back to the money-shot line plus the rung-2 regret number
(0.161 → 0.043) and the rung-3 cost number (48 vs 72) — those carry the story alone.
