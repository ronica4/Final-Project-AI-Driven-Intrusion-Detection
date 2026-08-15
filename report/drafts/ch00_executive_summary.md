# Executive Summary

DNS exfiltration (MITRE T1048.003/T1071.004/T1572) tunnels stolen data through DNS queries/subdomain
labels — traffic every network must permit, hard to firewall. Two vantage points: **Dataset A**
(CIC-Bell-DNS-EXF-2021, plaintext resolver logs, stateless per-query features) and **Dataset B**
(CIRA-CIC-DoHBrw-2020, encrypted DoH flow telemetry). A unified 11-column schema spans both.

**Four architectures, both datasets.** XGBoost and 1D-CNN: strong on B (F1≈0.99–1.00) but capped at
F1≈0.82 on A by a ~40% FPR wall no threshold tuning removes. Isolation Forest and Autoencoder: fail on
both, worse on A (ROC-AUC below chance, 0.26/0.264) — exfiltration isn't a minority-density anomaly on
either dataset.

**Production base-rate honesty.** At realistic ~0.01% prevalence, Dataset A's system generates ≈4M false
alerts/day for ≈1,000 true ones (≈0.025% precision); Dataset B hard framing stays SOC-viable
(≈1,515 false alerts/day, ≈40% precision) — the sharpest quantitative contrast in this report.

**Cascade (Isolation Forest → XGBoost → escalation).** F1=0.5053 vs. standalone XGBoost's 0.9999 —
loses to its own majority-class baseline (0.6667). Diagnosis: Stage 1 is irreversible, so the cascade's
recall ceiling equals Stage 1's own recall; Isolation Forest has no principled way to prefer either half
of a balanced, bimodal distribution.

**Cross-dataset transfer collapses** in 7 of 8 cells, all four models, to a fixed-guess classifier
(KS 0.89–1.00 on every shared feature; `vol_total` differs ~1,859× in raw scale despite being predicted
the most transferable family). Verdict: distribution shift, not overfitting — in-domain scores stay
strong, and one cell (CNN, B→A) keeps ROC-AUC=0.773 despite F1=0.0000, meaning the signal survives
transfer and only the threshold miscalibrates. Deployment implication: train and calibrate separately
per vantage point.
