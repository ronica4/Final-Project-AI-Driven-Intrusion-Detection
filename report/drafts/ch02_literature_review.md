# Chapter 2 — Literature Review

Two papers anchor this project's feature design and are contrasted directly below, since one supplied
the dataset schema and the other supplied the threat model that dataset's heavy/light split
operationalizes.

- **Nadler, A., Aminov, A., & Shabtai, A. (2019).** "Detection of Malicious and Low Throughput Data
  Exfiltration Over the DNS Protocol." *Computers & Security*, 80, 36–53.
  DOI: 10.1016/j.cose.2018.09.006.
- **Mahdavifar, S., et al. (2021).** "Lightweight Hybrid Detection of Data Exfiltration using DNS based
  on Machine Learning." *Proceedings of the 2021 11th International Conference on Communication and
  Network Security (CNS '21)*, ACM. DOI: 10.1145/3507509.3507520. — this is the paper introducing the
  **CIC-Bell-DNS-EXF-2021** dataset used as "Dataset A" throughout this project.

## 2.1 Extraction matrix

| Axis | Nadler et al. (2019) | Mahdavifar et al. (2021) |
|---|---|---|
| **Entropy, semantically** | A *window-level statistical anomaly signal*: character-distribution statistics of a client's queries to a domain are aggregated over a sliding time interval and compared against a learned per-client/domain baseline. Entropy is one input to a distributional-shift judgment, not a per-record score. | A *direct per-record feature*: Shannon entropy of a single query's encoded label is computed once and fed straight into a supervised classifier alongside other stateless/stateful features — no baseline or window required. |
| **Per-query or per-session** | **Per-session/per-window.** The detector scores a (client, domain) pair after aggregating its query history over time — this is what lets it catch an attacker deliberately throttling throughput below any single-query threshold. | **Per-query, plus limited per-flow stateful aggregates.** The architecture is explicitly "two-layer": stateless features computed on each query/packet directly, augmented with a modest set of stateful features (inter-arrival time statistics, flow duration) — but with no long-horizon client-level windowing. |
| **Detection paradigm** | Supervised *feature selection* feeding an interchangeable, adjustable **anomaly-detection** model — reported ≥99% recall / <0.01% FPR for tunneling, but explicitly flags low-throughput exfiltration (payload throttled to ≤1 KB/h under DNS syntax limits) as the harder case. | Fully **supervised classification** — five algorithms compared (GNB, RF, MLP, SVM, LR) against ground-truth heavy/light/benign labels; Random Forest reported as the best performer. |

## 2.2 What we adopted, modified, or rejected — and why

**Adopted.**
- *Entropy as a core signal (F2).* Both papers independently converge on encoding randomness as
  discriminative, despite scoring it completely differently (window-anomaly vs. per-record). That
  convergence from two different methodologies is why F2 is one of this project's three
  encryption-surviving INTERSECTION families rather than a Dataset-A-only feature.
- *Volume/size features (F1).* Mahdavifar's stateless packet-size and query-length features map
  directly onto this project's `vol_primary`/`vol_secondary`/`vol_total`. Since Mahdavifar's paper is
  the direct source of Dataset A, this is close to a literal adoption rather than a reinterpretation.
- *Supervised classification as the primary detection paradigm*, matching Mahdavifar rather than
  Nadler — this project has ground-truth attack/benign labels on both datasets, and the rubric calls
  for a comparison across supervised ML/DL architectures.

**Modified.**
- *Nadler's time-windowed, per-client-aggregated anomaly framing* is not built for Dataset A (each row
  there is an independent per-query record with no client/session identifier to window over — see
  §2.3), but the same underlying signal reappears in a different form as this project's **F4 temporal
  rhythm** family on Dataset B, where DoH flow-level features (`PacketTimeMean`,
  `PacketTimeStandardDeviation`, `PacketTimeSkewFromMedian`) give an inter-arrival-rhythm signal that is
  structurally much closer to Nadler's windowed statistics than anything Dataset A can supply. This
  family is therefore B_ONLY by construction, not by oversight (Chapter 5, decision D2).

**Rejected.**
- *Nadler's unsupervised anomaly-detection model* as the *primary* detector — kept only as the
  cascade's first stage (Isolation Forest, Step 2E), where its cheap, high-recall, false-positive-tolerant
  profile is actually the right fit, rather than as the project's main classifier.
- *Nadler's per-client/per-domain long-horizon windowing*, entirely, for Dataset A: CIC-Bell-DNS-EXF-2021
  ships as independent per-query rows with no session key that would let this project reconstruct such
  windows without fabricating one. Recorded as a limitation, not silently worked around.

## 2.3 The Nadler → light-class lineage, and where our result diverges from it

Nadler et al.'s central contribution is identifying that an attacker who deliberately throttles DNS
exfiltration throughput evades detectors built around volume thresholds — precisely because low
throughput makes individual queries statistically resemble normal traffic. Mahdavifar et al. operationalize
that exact threat model as CIC-Bell-DNS-EXF-2021's **light_attack** subclass (vs. **heavy_attack**), and
this project inherits that split directly (Chapter 1.5).

The literature's expectation, carried forward into this project's original plan, was therefore: *light
attacks should be materially harder to detect than heavy attacks* — a light-class recall gap should be
the headline forensic finding of Chapter 8. **Step 3A's real result on Dataset A does not bear this out**:
light-class recall (99.95%) and heavy-class recall (99.94%) are statistically indistinguishable, and the
actual operational bottleneck is false positives (40.5% FPR), not missed light attacks. This is reported
in Chapter 8.1 as a genuine divergence from the Nadler-derived hypothesis rather than reframed to match
it — on the F1–F3 feature set this project measures, the throughput-stealth tradeoff Nadler identified
does not translate into a light-class blind spot the way the prior literature would predict. That
divergence is itself a finding worth stating plainly, not a failure to replicate.
