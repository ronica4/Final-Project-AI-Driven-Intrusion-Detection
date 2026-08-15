# Chapter 2 — Literature Review

Two anchor papers: one supplied the dataset schema, the other the threat model its heavy/light split
operationalizes.

- **Nadler, A., Aminov, A., & Shabtai, A. (2019).** "Detection of Malicious and Low Throughput Data
  Exfiltration Over the DNS Protocol." *Computers & Security*, 80, 36–53. DOI: 10.1016/j.cose.2018.09.006.
- **Mahdavifar, S., et al. (2021).** "Lightweight Hybrid Detection of Data Exfiltration using DNS based on
  Machine Learning." *CNS '21*, ACM. DOI: 10.1145/3507509.3507520 — introduces **CIC-Bell-DNS-EXF-2021**
  (Dataset A).

## 2.1 Extraction matrix

| Axis | Nadler et al. (2019) | Mahdavifar et al. (2021) |
|---|---|---|
| Entropy | Window-level anomaly signal, char-distribution stats vs. learned baseline | Per-record Shannon entropy fed straight into a classifier |
| Scope | Per-session/window — catches throttled throughput | Per-query + limited aggregates, no long-horizon windowing |
| Paradigm | Feature selection → anomaly detection; ≥99% recall/<0.01% FPR, low-throughput (≤1 KB/h) harder | Fully supervised — 5 algorithms compared, RF best |

## 2.2 Adopted, modified, rejected

**Adopted:** entropy (F2) — both papers converge independently, hence INTERSECTION; volume/size (F1) —
Mahdavifar's stateless features map directly onto `vol_*`; supervised classification as primary paradigm.

**Modified:** Nadler's time-windowed framing isn't buildable on A (no session ID), reappears as this
project's **F4 temporal rhythm** on B — B_ONLY by construction (Ch. 5, D2), not oversight.

**Rejected:** Nadler's unsupervised model as primary detector — kept only as cascade Stage 1 (Isolation
Forest); Nadler's long-horizon windowing, entirely, for A.

## 2.3 Where our result diverges

Nadler: throttled exfiltration evades volume-threshold detectors. Mahdavifar operationalizes this as
**light_attack** vs. **heavy_attack** (Ch. 1.4), predicting light attacks are materially harder to detect.
**Step 3A does not bear this out** — light (99.95%) and heavy (99.94%) recall are statistically
indistinguishable; bottleneck is false positives (40.5% FPR), not missed light attacks (Ch. 8.1) — a
genuine divergence from the Nadler-derived hypothesis, stated plainly.
