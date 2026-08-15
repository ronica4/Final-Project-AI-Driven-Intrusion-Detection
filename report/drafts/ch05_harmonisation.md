# Chapter 5 — Harmonisation Across the Encryption Boundary

## 5.1 Unified schema specification

Every loader emits the same 11 columns, same order, grouped into five behavioural families
(`schema/unified.py`). Column order matters — the 1D-CNN's receptive field (Ch. 6.1) only works because
related features sit adjacent.

| Family | Columns | Role | Security meaning |
|---|---|---|---|
| **F1** Payload volume | `vol_primary/secondary/total` | INTERSECTION | Exfiltration pushes more bytes than normal |
| **F2** Encoding randomness | `rand_entropy`, `rand_dispersion` | INTERSECTION | Encoded binary looks statistically random |
| **F3** Structural complexity | `struct_segments`, `struct_max_segment` | INTERSECTION | Chunking produces unusual names |
| **F4** Temporal rhythm | `time_central/dispersion/skew` | B_ONLY | Automated exfiltration has regular timing |
| **F5** Endpoint dispersion | `disp_uniqueness` | B_ONLY | Traffic concentrates on few endpoints |

F1–F3 have a genuine, if imperfect, counterpart on both sides; F4–F5 exist only in B (A is restricted to
*stateless* per-query features, D1). B's F2/F3 are statistical proxies for, not equivalents of, A's direct
string measures — motivating the ablation below.

## 5.2 Cross-dataset transfer, ablation, distribution shift

Run against real data, all four models (`evaluation/cross_dataset.py`). **Headline: transfer collapses to a
trivial classifier in 7 of 8 cells, all four models.** In-domain (5-fold CV) vs. transfer
(fit-on-source/score-on-target), `mode="intersection"` (7 shared columns), scaler fit on source only
(honest-deployment simulation).

| model | A→A | B→B | A→B (transfer) | B→A (transfer) |
|---|---|---|---|---|
| XGBoost | F1=0.8182 | F1=0.9999 | F1=0.6667, FPR=1.0, AUC=0.707 | F1=0.0000, AUC=0.490 |
| Isolation Forest | F1=0.1051 | F1=0.3954 | F1=0.6667, FPR=1.0, AUC=0.579 | F1=0.6460, FPR=1.0, AUC=0.379 |
| CNN | F1=0.8176 | F1=0.9927 | F1=0.6667, FPR=1.0, AUC=0.708 | F1=0.0000, **AUC=0.773** |
| Autoencoder | F1=0.0474 | F1=0.6773 | F1=0.6667, FPR=1.0, AUC=0.671 | F1=0.6460, FPR=1.0, AUC=0.359 |

In-domain cells match each model's own single-dataset result. Every transfer cell collapses to (or near) a
trivial fixed-guess: F1=0.6667/FPR=1.0 always-positive for B (rate≈0.50); F1=0.6460 same for A (0.4772);
F1=0.0000 always-negative. **No model, of four structurally different, produces a working cross-dataset
detector.** Exception: CNN's B→A cell, F1=0.0000 but **AUC=0.773** — ranking survives transfer; it simply
lands a threshold (calibrated on B's scale) that calls zero of A's rows positive.

**Ablation (D3) — falsified unanimously.** `families="F1_only"` vs. `intersection`: F1-only transfers
better in **zero of eight cells** (mostly tied; XGBoost A→B strictly worse, 0.0000 vs. 0.6667). Payload
volume alone does not survive the encryption boundary intact.

**Distribution shift.** KS per intersection column (full table Appendix D.3): every column 0.89
(`rand_entropy`) to 1.0000 (`vol_primary/secondary/total`, `struct_segments`) — near-total or fully
disjoint support. `vol_total` differs ~1,859× — A's realisation (`FQDN_count`, small integer) and B's
(`FlowBytesSent`, cumulative flow bytes) are behaviourally analogous but not numerically comparable,
contradicting the premise that F1 is "the family with a genuinely direct counterpart." Z-scoring
(source-only fit) doesn't repair this — it preserves each dataset's distribution *shape*, and the shapes
differ enough that a boundary learned in one standardised space doesn't carve the other the same way.
**Not tried, given the deadline:** `log1p` on F1 before scaling, or a jointly-fit scaler.

## 5.3 Scaling remedies

Inside `build_pipeline()`: **z-scoring within dataset** (fit per-fold/source only, transfer stays honest);
**`log1p` on heavy-tailed volume features** (`rand_dispersion`'s B realisation is
`log1p(PacketLengthVariance)`); **median imputation** (`keep_empty_features=True` — A's all-NaN B_ONLY
columns imputed to a constant rather than silently dropped, so "no signal here" becomes structural).

## 5.4 The observability finding (D2)

**Three of five behavioural families are unobservable, or observable only as a weak proxy, once traffic
crosses the plaintext-to-encrypted boundary** — this project's central empirical claim. F4/F5 entirely
unobservable on A (100% NaN by construction, no session key). F2/F3 survive only as reconstructed proxies
(§5.2 ablation). **F1 was predicted the one family with a genuinely direct counterpart — that did not
hold:** its columns are among the *most* distributionally shifted of the intersection set. "Computable on
both sides" ≠ "means the same numeric thing on both sides"; F1 satisfies only the first, no better than
F2/F3.

**Practical implication:** an operator with only encrypted DoH captures retains conceptual visibility into
volume, randomness, structural complexity, zero visibility into rhythm or fan-out — but visibility ≠
**transferability**. Takeaway: **train and calibrate separately per vantage point.**
