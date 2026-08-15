# Executive Summary

DNS exfiltration abuses name resolution as a covert channel: a compromised host encodes stolen data into
subdomain labels (or, over DoH, into request/response byte patterns) and emits a stream of lookups that
must survive firewalls built to allow DNS through by default. This maps to MITRE ATT&CK **T1048.003**
(exfiltration over an unencrypted non-C2 protocol) and, on the encrypted side, **T1071.004**/**T1572** —
the same attack goal, observed from two different vantage points with very different visibility into it.

This project builds and evaluates detectors against both vantage points: **Dataset A**
(CIC-Bell-DNS-EXF-2021, plaintext DNS query logs, stateless per-query features) and **Dataset B**
(CIRA-CIC-DoHBrw-2020, encrypted DNS-over-HTTPS flow captures, packet-shape statistics). A five-family
feature schema (payload volume, encoding randomness, structural complexity, temporal rhythm, endpoint
dispersion) was designed to be conceptually comparable across both datasets, with three families
(F1–F3) computable on both sides and two (F4–F5) observable only in Dataset B's richer flow-level
telemetry.

**Four architectures were trained and evaluated on both datasets: XGBoost, Isolation Forest, a 1D-CNN,
and an Autoencoder.** The two supervised models (XGBoost, CNN) perform almost identically to each other
on each dataset — strong on Dataset B (F1≈0.99–1.00, and XGBoost's Dataset B run produced **zero false
positives across 19,807 held-out benign rows**), weaker on Dataset A (F1≈0.82, with recall above 99.8%
but a persistent ~40% false-positive rate that threshold tuning could not fix — Chapter 8's core
diagnostic result). The two unsupervised models (Isolation Forest, Autoencoder) fail on Dataset A
specifically — both score **below-chance ROC-AUC**, a result triangulated two independent ways, because
exfiltration traffic there is not a minority density the way both architectures' core assumption
requires. Extrapolated to a realistic production base rate (~1:10,000), this gap has real operational
consequences: Dataset A's XGBoost would generate on the order of **four million false alerts a day** for
roughly 1,000 true detections, while Dataset B's, under the same assumptions, would run at close to **40%
analyst-facing precision** — the feature set, not the base-rate arithmetic itself, is what determines
whether a detector is production-viable.

**A three-stage hybrid cascade (Isolation Forest → XGBoost → escalation) was built and tested honestly: it
does not beat standalone XGBoost.** On Dataset B's hard framing the cascade scores F1=0.505 against
XGBoost's 0.9999, bottlenecked entirely by Isolation Forest's weak first-stage recall (33%) — a
diagnosable, reported-as-is negative result rather than a reframed one, and a finding that itself argues
for the cascade's premise needing a stronger anomaly-detection stage before this architecture is usable.

**The project's central scientific finding concerns what survives the encryption boundary.** A
cross-dataset transfer experiment — train on one vantage point, deploy on the other, without retraining —
was run across all four models. **Transfer collapses to a trivial fixed-guess classifier in 7 of 8 cases
tested, regardless of architecture.** Distribution-shift analysis explains why directly, not by inference:
every one of the three shared feature families shows near-total numerical divergence between datasets
(Kolmogorov–Smirnov statistics of 0.89–1.00), **including payload volume — the one family originally
predicted to be scale-stable across the boundary, and the family whose own raw values instead differ by
up to three orders of magnitude between datasets.** The corrected reading: a feature family being
*computable* on both sides of an encryption boundary does not mean it is *numerically equivalent* there.
Two DNS-exfiltration detectors, however similarly named their inputs, must be trained and calibrated
separately per vantage point — one model does not transfer to the other, and this project's four
independently-trained architectures all confirm it the same way.
