# Chapter 8.2 — Cross-Dataset Comparison: The Verdict

## 8.2.1 The question

Ch. 5 (§5.2) measured transfer collapsing to a trivial fixed-guess classifier in 7 of 8 cells, all four
models. **Distribution shift** (environments genuinely differ, fixed by retraining per environment) or
**overfitting** (model memorised training idiosyncrasies, meaning it never learned the intended signal)?

## 8.2.2 The verdict: distribution shift, not overfitting

1. **In-domain performance strong for every model on both datasets**, ruling out "never learned a real
   signal": XGBoost/CNN F1≈0.82 on A, F1≈0.99–1.00 on B, recall >0.99 both ways — failure is specific to
   *crossing* datasets.
2. **§5.2.3 measures the cause directly** — every intersection column KS 0.89–1.00; `vol_total` differs
   ~1,859× in raw scale, independent of model behaviour.
3. **Failure mode matches shift's signature, not overfitting's.** An overfit model degrades gracefully or
   inconsistently; observed is uniform collapse to a **fixed decision** (FPR/recall pinned at 1.0 or 0.0
   in 6 of 8 cells) — a boundary learned in one standardised space landing entirely on one side of the
   other's.
4. **One partial exception reinforces this.** CNN's B→A cell: F1=0.0000 but ROC-AUC=0.773 — an overfit
   model's ranking should be no better than random once transferred; it isn't. Signal survived; only the
   fixed 0.5 threshold, calibrated on B's scale, failed to land sensibly.

## 8.2.3 Per-cell diagnosis

| direction | model | outcome | what it demonstrates |
|---|---|---|---|
| A → B | all four | always-positive (F1≈0.65–0.67, FPR=1.0) | A-trained boundary reads all of B's rescaled inputs as "high enough" |
| B → A | XGBoost, CNN | always-negative (F1=0.0000) | B-trained boundary reads all of A's inputs as "too low" |
| B → A | IsoForest, AE | always-positive (F1≈0.65, FPR=1.0) | opposite collapse — benign-shaped reference from B treats A as anomalous |
| B → A | CNN (ranking) | AUC=0.773 despite F1=0.0000 | signal survives; only the threshold fails |

Same direction (B→A), opposite fixed-classifier outcomes by model family — rules out a single explanation
("B's data is just numerically larger"), pointing to each model's own decision function interacting with
the scale mismatch.

## 8.2.4 What this means for deployment

None of the four architectures can be trained once and deployed across both a plaintext-DNS log and an
encrypted DoH capture. The unified schema's abstraction is real at *what each family measures* — but
insufficient to make a model portable, since the numeric realisation differs by orders of magnitude
(§5.2.3). **Train and calibrate separately per vantage point** — two genuinely different measurement
regimes sharing a conceptual taxonomy, not stand-ins for "any DNS exfiltration data." Proposed remedies
(`log1p` on F1 before scaling, jointly-fit scaler) documented but not tested against this matrix; diagnosis
covers `mode="intersection"` only — B_ONLY families (F4/F5) can't be tested for transfer at all, by
construction.
