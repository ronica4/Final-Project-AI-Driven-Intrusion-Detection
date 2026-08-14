# Chapter 4 — Feature Ranking and the Leakage Demonstration

This is the strongest single narrative beat in the report: we deliberately cheat, catch ourselves,
and report the honest number — twice, once per dataset, via two structurally different leakage
mechanisms. **Dataset A's half (the `sld` demo) is complete and produced a genuine, unplanned finding.
Dataset B's half (the `SourceIP` demo) has not been run yet** — `data/dohbrw2020/` is not populated on
this machine, and it is a separate piece of work from the Dataset B EDA numbers Teammate B already
backfilled for Chapter 3. It is flagged **PENDING** at every point below rather than filled in with an
assumed or predicted number, consistent with this project's standing rule never to report a result that
hasn't actually been run.

## 4.1 Figure 4.1 — Gain-based feature importance

Gain-based importance (not the XGBoost default weight-based importance) is used deliberately: weight
counts how many times a feature is split on and is biased toward high-cardinality features — exactly the
kind of artifact this chapter's leakage demonstration exists to catch, so using it here would undercut
the chapter's own argument.

**Dataset A** (`runs/figures/exf2021_feature_importance.png`, clean run, `sld`/timestamp already
excluded by the production loader):

| feature | gain share |
|---|---|
| rand_dispersion | 0.5112 |
| vol_total | 0.2767 |
| struct_segments | 0.1046 |
| vol_secondary | 0.0728 |
| struct_max_segment | 0.0187 |
| vol_primary | 0.0105 |
| rand_entropy | 0.0055 |
| all 4 B_ONLY columns | 0.0000 |

The 4 B_ONLY columns receive exactly zero gain — XGBoost never splits on a post-imputation constant.
This is the same "effective input dimensionality is 7, not 11 on Dataset A" finding as Chapters 3 and 6,
now confirmed a third, independent way: from the model's own splitting behaviour rather than the NaN
audit or the significance table.

**Dataset B**: **PENDING** — not run locally.

## 4.2 Leakage demonstration #1 — Dataset A, `sld`

Found during the Step 0C header check, not in the original project spec: `sld` (second-level domain, raw
text) takes only **22–33 distinct values in attack traffic** (sample-dependent) but **11,134–40,599 in
benign traffic** — structurally the same class of trap as Dataset B's `SourceIP` column (§4.3):
low-cardinality identifiers that let a model memorise "which of a handful of fixed testbed values" a row
holds, instead of learning any real behavioural signal.

**Encoding note:** `sld` is text, not numeric, so it cannot be fed to XGBoost as-is. Two encodings were
tried, specifically to separate "did leakage happen" from "was it a labelling-scheme artifact":

1. **Label-encoded** (`pd.factorize`, arbitrary insertion-order integer codes) —
   `runs/metrics/leakage_demo_exf2021.json`.
2. **Binary "known-attack-`sld`" indicator** (1 if this row's `sld` value was ever seen in attack
   traffic, 0 otherwise) — collapses cardinality entirely, the closest possible analogue to a clean
   identity lookup — `runs/metrics/leakage_demo_exf2021_binary_variant.json`.

| | F1 (clean, `sld` dropped) | F1 (dirty, `sld` included) | dominant feature, dirty run |
|---|---|---|---|
| Label-encoded `sld` | 0.8182 | 0.8184 | `_leakage_sld`, 75.5% of gain |
| Binary known-attack indicator | 0.8182 | 0.8184 | `_leakage_sld`, 95.4% of gain |

**This is a genuine, unplanned finding, not a simple replication of the "expect ≈0.99+" prediction the
plan made by analogy to `SourceIP`.** Importance dominates exactly as expected in both encodings — but
score barely moves at all (+0.0002 F1), a dramatically muted effect compared to a `SourceIP`-style
memorised lookup. Running both encodings rules out the obvious alternative explanation (that
`factorize`'s arbitrary codes, scattered across ~40K distinct values, are too fine-grained for ordinal
tree splits to isolate cleanly) — the binary variant collapses that concern entirely and gets the
identical result.

**The actual explanation, found by checking class overlap on the `sld` value sets directly:** of the 33
`sld` values appearing anywhere in attack traffic in this run's sample, **30 also appear in benign
traffic** — only 3 are attack-exclusive. `sld`'s low attack-side cardinality is real (confirming Step
0C's original D11 finding), but unlike `SourceIP` (reportedly used exclusively by the attacker in
Dataset B's testbed — to be confirmed once §4.3 actually runs), `sld`'s low-cardinality value set is
mostly **shared** with benign traffic, not exclusive to attack. A "known-attack-`sld`" lookup is
therefore a noisy, weakly-predictive signal rather than a clean shortcut — which is exactly why the
model still leans on it heavily (it remains the most useful available split) without that translating
into a large score jump.

**This refines, rather than overturns, the leakage classification:** `sld` is correctly dropped by the
production loader — its cardinality skew is real and it is still a testbed-shaped artifact rather than a
generalisable behavioural signal — but it is demonstrably a structurally weaker leakage mechanism than a
`SourceIP`-style exclusive identifier, and the difference is now quantified rather than assumed by
analogy.

## 4.3 Leakage demonstration #2 — Dataset B, `SourceIP` — **PENDING**

**Not yet run.** Per the plan (Chapter 4 spec / Step 2C item 3), the expected shape of this demo is:

1. Train with `include_leakage_columns=True` (`SourceIP` retained) — expect F1/PR-AUC ≈ 0.99+, since the
   attacker in Dataset B's testbed reportedly used a single fixed IP throughout, making `SourceIP` a
   near-exact lookup table for the label.
2. Plot gain-based importance — expect `SourceIP` to dominate the ranking outright.
3. Re-run with the 5 identifier columns dropped (the production loader default) — report the honest
   F1/PR-AUC.
4. Two-row before/after table, same format as §4.2's table above, plus one paragraph naming it as a
   testbed artifact.

Whether it actually plays out this cleanly — a `SourceIP`-style *exclusive* lookup, unlike `sld`'s
*partial-overlap* one — is an empirical question, not something to assume in advance; §4.2 is the
cautionary example of why. **This section will be filled in once Teammate B runs the demo** (the code
path, `features/selection.py`'s `factorize_leakage_column()` / `gain_importance()` /
`plot_feature_importance()`, is dataset-agnostic and already built — this is a data-locality gap, not a
missing-code gap).

## 4.4 Multicollinearity — Variance Inflation Factor

VIF computed via `sklearn.linear_model.LinearRegression` (`VIF_i = 1/(1 − R²_i)`), not `statsmodels` —
the formula is direct enough that adding a new pinned dependency both teammates would need to install
wasn't worth it for one calculation.

**Dataset A**, all 11 columns (`runs/metrics/vif_exf2021.json`) — substantially more collinearity than
the plan anticipated (it predicted `vol_primary`/`vol_secondary` specifically as the one collinear pair):

| feature | VIF | flagged (>10) |
|---|---|---|
| vol_primary | 128.3 | yes |
| vol_secondary | 113.6 | yes |
| vol_total | 136.3 | yes |
| rand_entropy | 2.33 | no |
| rand_dispersion | 14.4 | yes |
| struct_segments | 100.5 | yes |
| struct_max_segment | 168.0 | yes |
| 4 B_ONLY columns | undefined (all-NaN) | — |

**6 of 7 testable columns are heavily collinear** (VIF ≫ 10) — nearly all of F1 (volume) and F3
(structure) are redundant with each other, consistent with Chapter 3's flagged correlation pairs
(`vol_secondary` ↔ `struct_segments`, `rand_dispersion` ↔ `struct_segments`). **Not acted on for
XGBoost** — tree splits are robust to collinearity, consistent with the model still using several of
these redundant columns productively in §4.1's gain ranking — but flagged explicitly for Chapter 6's
CNN/Autoencoder discussion, where dense/convolutional inputs are more sensitive to redundant features
than tree splits are.

`rand_entropy` has the **lowest** VIF of all 7 columns (2.33) — the least redundant feature in the set —
which sharpens rather than contradicts its near-zero gain importance in §4.1: it isn't being crowded out
by a correlated feature, it genuinely carries little independent signal on this dataset (corroborated a
third way by Chapter 3's "negligible" Cliff's delta for the same column).

**Dataset B**: **PENDING** — not run.

## 4.5 Discrepancy analysis — the three rubric questions, answered per surprise (Dataset A)

**1. Did the algorithm agree with our intuition as security analysts?** Partially. Volume and
encoding-randomness (`vol_total`, `rand_dispersion`) jointly dominating gain (~79% combined) matches the
prior that payload volume and randomness are the primary exfiltration signals. It did **not** expect the
specific, classically-cited entropy feature (`rand_entropy`) to rank last among testable features
(0.55% of gain).

**2–3. What surprised us, and is each surprise leakage / multicollinearity / a genuine latent pattern?**
Answered per surprise, not in general, because the three surprises below have three different causes:

| Surprise | Classification | Evidence |
|---|---|---|
| `rand_entropy`'s near-zero importance | **Genuine latent pattern** | VIF = 2.33, the *lowest* of all 7 (rules out multicollinearity as the cause); Cliff's δ = −0.125, "negligible" (Chapter 3, independently corroborates via a completely different method) |
| `sld` dominates importance but barely moves the score | **Neither straightforward leakage-as-predicted nor multicollinearity** — a refinement of the leakage finding itself | Class-overlap analysis: 30 of 33 attack `sld` values also appear in benign traffic (§4.2) |
| Near-universal high VIF across F1/F3 | **Multicollinearity**, confirmed directly | Consistent with Chapter 3's correlation heatmap; larger in scope than the plan predicted, not different in kind |

Two independent methods (univariate effect size in Chapter 3, multivariate gain importance here) agree
that `rand_entropy` underperforms its security-literature reputation on this specific dataset — a
finding worth stating plainly in Chapter 8's benchmark comparison (Chapter 8.3) rather than treating as
an implementation quirk.

## 4.6 What's still open

- Dataset B's `SourceIP` leakage demo (§4.3) and VIF table (§4.4) — pending Teammate B's local run;
  code is dataset-agnostic and ready.
- Once §4.3 lands, §4.5's discrepancy table should be extended with a fourth row comparing the two
  datasets' leakage mechanisms directly (exclusive identifier vs. partial-overlap identifier) — that
  comparison is likely to be a stronger Chapter 4 closer than either demo alone, matching Chapter 2's
  observation that this project's data-driven findings tend to be strongest when two independent methods
  or datasets corroborate (or productively disagree with) each other.
