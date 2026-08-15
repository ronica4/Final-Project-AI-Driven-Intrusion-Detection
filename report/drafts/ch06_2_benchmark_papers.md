# Chapter 6.2 — Architecture-Specific Literature Benchmarks

Four papers, one per model architecture used in this project, each applying that architecture to
DNS tunneling/exfiltration or DoH malicious-traffic detection specifically (general-network fallback
only where a DNS-specific paper with usable metrics could not be found). These numbers are collected
once here and reused for the benchmark comparison in Chapter 8.3, per this chapter's own instruction.

**A note on verification, in the interest of not overstating confidence:** each citation below was
checked against at least one full-text fetch; where a fetch was paywalled (ResearchGate, MDPI, Nature,
Springer, ScienceDirect all returned HTTP 403 or a login wall at various points) an alternate open
mirror (PMC, arXiv, a direct DOI resolve) was tried before falling back to a WebSearch-indexed
snippet. Two of the four entries below carry an explicit **"not independently verified against full
text"** flag where that alternate route still didn't clear the paywall — the citation (authors, year,
title, venue, DOI) is confirmed in every case; it is specifically the *numbers* that carry the caveat
in those two entries.

---

- **Abrahim, H., Hou, W., Zhuang, Y., & Rahman, H. U. (2026).** "Enhancing intrusion detection in
  encrypted DoH traffic through a robust ensemble learning framework." *PLOS ONE*.
  DOI: 10.1371/journal.pone.0345880
  Dataset: **CIRA-CIC-DoHBrw-2020** — the same dataset family as this project's Dataset B — and
  DoH-Tunnel-Traffic-HKD. Configuration: XGBoost as the **meta-learner** in a stacking ensemble over
  LSTM and GRU base learners, trained on out-of-fold predictions via stratified 5-fold CV;
  `max_depth`, `learning_rate`, and `n_estimators` were tuned, remaining hyperparameters left at
  library defaults. Reported (on CIRA-CIC-DoHBrw-2020, **ensemble-level** — this is the stacked
  ensemble's output with XGBoost as the final layer, not an XGBoost-alone ablation, which is a fair
  comparison to flag explicitly in Ch 8.3 against this project's standalone-XGBoost numbers):
  Recall=0.9996, Precision=0.9967, F1=0.9981.

- **Li, H., Li, Z., Zhang, S., & Pu, X. (2024).** "Malicious DNS detection by combining improved
  transformer and CNN." *Scientific Reports*. DOI: 10.1038/s41598-024-81189-1
  Dataset: DGA (~10,000 domain records, 60% malicious) and MDND (~10,000 domain records, multiple
  attack types), with application testing against **CIC-Bell-DNS** — the same dataset family as this
  project's Dataset A. Configuration: a 1D-CNN (first conv layer: kernel size 2, 64 filters → max
  pool, stride/window 3; second conv layer: kernel size 2, 32 filters → max pool, stride/window 3;
  flatten → 64-unit fully connected layer) processing traffic-information features, combined with an
  improved Transformer processing domain-name sequences, in a block-based ensemble ("ITransformer-CNN").
  Reported: Accuracy=95.67%, Recall=83.21%, F1=88.43%. The paper does not report Precision or FPR as
  separate figures in its main results table.

- **Wang, S., Sun, L., Qin, S., Li, W., & Liu, W. (2022).** "KRTunnel: DNS channel detector for mobile
  devices." *Computers & Security*, 120, 102818. DOI: 10.1016/j.cose.2022.102818
  Dataset: DNS request/response traffic collected for mobile-device DNS tunnel detection. Configuration:
  Isolation Forest as the classification stage of the KRTunnel pipeline, over a feature set extracted
  from DNS request/response pairs. Reported: **accuracy 98.1% on unseen DNS tunnel traffic**; in one
  comparison context the iForest model is reported to produce as few as ~2.5 false positives per day.
  **Not independently verified against full text** — ScienceDirect returned HTTP 403 on every fetch
  attempt (direct URL and DOI resolve both), so these two figures come from indexed search snippets
  rather than the paper itself, and a clean Recall/FPR/F1 triple could not be confirmed. Flagged rather
  than fabricated to fill the gap.

  *Fallback candidate consulted for the same slot, general-network rather than DNS-specific, but with a
  fully fetch-confirmed metric:* **Priyanshu, A., Shastri, S., & Medicherla, S. S. (2022).**
  "ARLIF-IDS — Attention-Augmented Real-Time Isolation Forest Intrusion Detection System." 43rd IEEE
  Symposium on Security and Privacy (poster session). arXiv:2204.09737. Dataset: NSL-KDD and
  KDDCUP'99. Configuration: Isolation Forest as the core detector, augmented with an attention
  mechanism, oriented toward real-time detection. Reported: F1=0.93 (averaged across both datasets);
  Recall, Precision, and FPR are not disclosed in the accessible abstract/content.

- **De Bernardi, G., Gaggero, G. B., Patrone, F., Zappatore, S., Marchese, M., & Mongelli, M. (2025).**
  "Rule-Based eXplainable Autoencoder for DNS Tunneling Detection." *Computers*, 14(9), 375.
  DOI: 10.3390/computers14090375
  **Citation confirmed** via two independent search sources (authors, year, title, venue, DOI all
  cross-checked and consistent). **Dataset, architecture configuration, and Recall/FPR/F1 numbers could
  not be verified** — every fetch attempt (the MDPI landing page, the MDPI `/htm` full-text variant, and
  a ProQuest-hosted PDF mirror) returned either HTTP 403 or no extractable body text. Reported here as a
  citation-only entry with the content gap stated plainly rather than filled with invented numbers; if
  Ch 8.3's benchmark table needs a real Autoencoder comparison point, either revisit this paper with a
  different access route (e.g. an institutional proxy) or substitute a different Autoencoder-for-DNS
  paper before that section is finalised.
