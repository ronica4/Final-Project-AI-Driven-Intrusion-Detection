# Chapter 8.2b — The base-rate honesty paragraph

> Status: self-contained, complete. Uses only Dataset A numbers already measured in Steps 2D/3A/3B —
> no dependency on Teammate B's remaining models. This slots into Chapter 8 alongside §8.1/§8.2 once
> those are assembled (§8.2 itself is still blocked on Step 2G).

## None of this project's framings reflect a realistic deployment base rate

Every number reported elsewhere in this project — F1, precision, recall, FPR — is measured against a
dataset where attack traffic is a substantial fraction of all rows, because that is what a supervised
training set needs to be usable at all:

| Framing | Positive rate |
|---|---|
| Dataset A (single framing) | 47.72% |
| Dataset B, hard framing | 50.00% (balanced by construction) |
| Dataset B, easy framing | 21.41% |
| **Real-world DoH/DNS traffic** | **~0.01% (≈ 1 in 10,000), by any reasonable operational estimate** |

None of these framings is a stand-in for production. This has to be said plainly rather than left as an
implicit caveat: a detector that looks excellent at 21–50% positive rate is being evaluated on a
question production will never actually ask it, because production traffic is overwhelmingly benign and
this project's datasets — like virtually every public IDS dataset — cannot supply that ratio without
either an unusably tiny attack count or a synthetically inflated one.

## What our measured FPR actually does at a realistic base rate

Take Dataset A's real, measured XGBoost numbers (Steps 2D/3B) and extrapolate them onto a plausible
production scale: **10 million DNS/DoH queries a day**, with attacks at the ~1:10,000 base rate above.

- Attacks/day: 10,000,000 × 0.0001 = **1,000**
- Benign/day: 10,000,000 − 1,000 = **9,999,000**

| | Before optimisation (t = 0.50) | After optimisation (t = 0.70) |
|---|---|---|
| Measured recall | 99.95% | 98.03% |
| Measured FPR | 40.48% | 39.72% |
| True positives caught / day | ≈ 1,000 | ≈ 980 |
| **False alerts / day** | **≈ 4,048,000** | **≈ 3,972,000** |
| Total alerts / day | ≈ 4,049,000 | ≈ 3,973,000 |
| Analyst-facing precision | ≈ 0.025% (1 real alert in ~4,050) | ≈ 0.025% (1 real alert in ~4,052) |

**Report this arithmetic honestly, including that it is unflattering.** At this project's measured FPR,
an analyst reviewing the model's output would see roughly **four million false alarms a day** for every
~1,000 real attacks it actually catches — a queue in which a genuine detection is buried under
thousands of false ones. This is not a niche edge case of the arithmetic; it is what a ~40% FPR *means*
at any base rate close to reality, regardless of how good recall looks in isolation.

**For contrast**, a hypothetical, well-calibrated detector at a 1% FPR — the kind of number IDS papers
often present as "acceptable" — would still generate 9,999,000 × 0.01 ≈ **100,000 false alerts a day**
against the same ~1,000 real attacks: about 100 false alarms per real detection. That illustrative
baseline is itself a lot for a SOC to absorb — and this project's actual system runs at roughly **40×**
that FPR, i.e. roughly 40× that alert volume. Put plainly, in the plan's own words: this is not a
detector at that operating point, it is a denial-of-service against your own SOC.

**One more honest observation, tying back directly to Step 3B:** the "before" and "after" columns above
are nearly identical. That is not a mistake in this extrapolation — it is the same finding Step 3B
already reported in miniature (raising the decision threshold moved FPR by less than one percentage
point, 40.48% → 39.72%, while costing real recall). At production scale, that means the threshold
optimisation attempted in Step 3B — while the right experiment to run, and honestly reported as a
negative result there — does **not** meaningfully change the SOC-queue picture on its own. The real
implication is structural, not a threshold-tuning problem: Dataset A's F1–F3 feature set produces a
probability distribution (Step 3B's near-step-function finding) that no single threshold choice can fix,
which is the concrete, quantified reason a cascade (Chapter 8.4) — rather than a better-tuned single
model — is the right response to this dataset's false-positive problem.

## What this section still needs before final assembly

- Dataset B's equivalent FPR (hard framing) is not yet in this draft — the tracker only records
  F1 ≈ 0.9999 for B's backfilled XGBoost run, not the underlying precision/recall/FPR breakdown needed
  to redo this same extrapolation for Dataset B. **Pulling the exact FPR from
  `runs/metrics/xgboost_dohbrw2020_hard.json` and adding a second Dataset-B column to the table above is
  the one remaining piece** — likely a very different story given B's near-perfect separation, and worth
  stating as a direct contrast to Dataset A's result once the number is in hand.
- This section assumes a single illustrative production scale (10M queries/day) and a single assumed
  base rate (1:10,000), matching the plan's own stated assumption — both are explicitly labelled as
  illustrative, not measured, and should stay labelled that way in the final chapter text.
