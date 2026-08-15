# Chapter 8.3 — Benchmark Comparison Against Published Results

Four papers, one per model architecture (Chapter 6.2), each applied to DNS tunneling/exfiltration or DoH
malicious-traffic detection. This section compares our real, measured numbers against each — attributing
discrepancies to specific, named causes rather than a vague "different setup," and reporting plainly
where our numbers are worse, where they are better, and where a real comparison is not possible at all.

## 8.3.1 XGBoost vs. Abrahim et al. (2026)

| | Recall | Precision | F1 |
|---|---|---|---|
| Abrahim et al. — stacking ensemble, XGBoost as meta-learner, CIRA-CIC-DoHBrw-2020 | 0.9996 | 0.9967 | 0.9981 |
| **This project — standalone XGBoost, Dataset B hard framing** | **0.9998** | — | **0.9999** |
| This project — standalone XGBoost, Dataset A | 0.9995 (t=0.50) | 0.6925 | 0.8182 (t=0.50) |

**Our standalone XGBoost on Dataset B slightly exceeds their full stacking ensemble's reported numbers**
— a real result, not a rounding artifact, but one that needs its comparison caveat stated plainly rather
than claimed as "we beat the state of the art": Abrahim et al.'s number is the **output of a
stacking ensemble** (XGBoost sitting as meta-learner atop LSTM and GRU base learners, trained on
out-of-fold predictions), not an XGBoost-alone ablation — the two numbers are not measuring the same
system, even though both are labelled "XGBoost" in the respective papers' headline tables. It is also
possible their combined training corpus (CIRA-CIC-DoHBrw-2020 **and** DoH-Tunnel-Traffic-HKD, per Ch 6.2)
differs enough from this project's DoHBrw-2020-only "hard" framing to explain part of the gap either way.
**The plan's own predicted pattern — "our hard-framing number will likely look worse than published
results, because published results often use the easy framing" — does not hold here.** Dataset B's hard
framing already sits at F1≈0.9999, at or above a published ensemble's number; there is no headroom left
for an easier framing to look better against this paper. (Dataset B's easy-framing XGBoost run was still
in progress on Teammate B's machine as of 15 Aug — not included here, and not needed for this specific
comparison since hard framing already meets/exceeds the benchmark.)

Dataset A's XGBoost number (F1=0.8182) is **not directly comparable** to Abrahim et al. at all — their
paper targets Dataset B's dataset family (CIRA-CIC-DoHBrw-2020) specifically, not Dataset A's
(CIC-Bell-DNS-EXF-2021). It is included here for completeness, not as a benchmarked comparison.

## 8.3.2 CNN vs. Li et al. (2024)

| | Accuracy | Recall | F1 |
|---|---|---|---|
| Li et al. — ITransformer-CNN block ensemble, CIC-Bell-DNS (Dataset A's family) | 95.67% | 83.21% | 88.43% |
| **This project — standalone 1D-CNN, Dataset A** | — | **99.80%** | **81.77%** |

**Our recall is dramatically higher (99.80% vs. 83.21%); our F1 is somewhat lower (81.77% vs. 88.43%).
The mechanism is diagnosable, not a mystery:** this project's CNN carries a high false positive rate on
Dataset A (FPR=40.44%, identical to the ~40% wall every supervised model in this report hits on this
dataset — Ch 8.1) that Li et al.'s reported metrics do not disclose (their paper reports Accuracy/Recall/
F1 only, no Precision or FPR, so a full apples-to-apples reconstruction of *their* precision is not
possible from the published numbers). Two structural differences also matter: (1) Li et al.'s model is
an **ITransformer-CNN block ensemble** — a Transformer processing domain-name character sequences fused
with a CNN over traffic-information features — not a standalone CNN over an 11-column feature vector like
this project's; the Transformer branch likely carries most of the precision gain from sequence-level
pattern matching this project's feature-only CNN has no access to. (2) Their evaluation corpus (DGA and
MDND datasets, with CIC-Bell-DNS used for **application testing** rather than as the primary training
set) is not confirmed to be the identical DNS-EXF-2021 attack corpus this project trains and tests on,
despite sharing the "CIC-Bell-DNS" family name — the same caveat Chapter 5 raised about dataset-family
names implying more equivalence than actually holds. **The honest reading: this project's CNN trades
precision for near-total recall on Dataset A, consistent with every other model tested on this dataset
(Ch 8.1's FPR wall), while Li et al.'s hybrid architecture — with access to sequence-level information
this project's schema deliberately does not expose (D1, stateless per-query features only) — achieves a
more balanced operating point.** This is a real, structural gap in what information the two systems have
access to, not primarily a training or tuning difference.

## 8.3.3 Isolation Forest vs. Wang et al. (2022, KRTunnel)

| | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Wang et al. — KRTunnel, Isolation Forest stage, mobile DNS tunnel traffic (**not independently verified against full text**) | 98.1% | — | — | — |
| **This project — Isolation Forest, Dataset A** | — | 7.45% | — | **0.1050** |
| **This project — Isolation Forest, Dataset B hard framing** | — | 23.91% | 59.51% | **0.3411** |

**This is the starkest gap in the entire benchmark comparison, and it is reported plainly rather than
explained away.** Wang et al.'s 98.1% accuracy figure is not independently verified (Ch 6.2 — every
fetch attempt returned HTTP 403; the number comes from an indexed search snippet, not the paper itself),
and accuracy alone is a weak metric under class imbalance — their paper does not disclose a
Recall/Precision/F1 triple this project can compare against directly. But even granting real uncertainty
about their exact number, **the direction of the gap is not in question: this project's Isolation Forest
performs far below a usable detector on both of this project's datasets**, and that finding has an
already-established, independently-diagnosed cause that has nothing to do with benchmark comparison at
all — **Step 2E's core result, confirmed a second way by the Autoencoder (§8.3.4) and a third way in
Chapter 5's transfer collapse: on both of this project's datasets, exfiltration/tunnel traffic is not a
minority density relative to benign traffic** (Dataset A's Isolation Forest even scores ROC-AUC=0.2610,
*below* chance). KRTunnel's mobile-device deployment context is plausibly a genuine low-base-rate,
minority-anomaly setting — exactly the premise Isolation Forest requires and exactly the premise this
project's two datasets (deliberately built with substantial attack proportions for supervised-model
statistical power, D5/D8) do not supply. **The gap is best read as evidence about this project's specific
datasets' density structure, not as a general indictment of Isolation Forest as an architecture** — a
distinction worth making precisely because conflating the two would understate what Chapters 5 and 8.1
already independently established, and overstate what this one comparison alone can support given the
unverified benchmark number.

## 8.3.4 Autoencoder vs. De Bernardi et al. (2025)

| | Recall | Precision | F1 |
|---|---|---|---|
| De Bernardi et al. — Rule-Based eXplainable Autoencoder for DNS Tunneling Detection (**citation confirmed; dataset, configuration, and all numbers unverified**) | — | — | — |
| **This project — Autoencoder, Dataset A** | 2.34% | 30.06% | **0.0434** (ROC-AUC=0.2643, below chance) |
| **This project — Autoencoder, Dataset B hard framing** | 26.81% | 84.00% | **0.4064** (below majority baseline 0.6667) |

**No real numeric comparison is possible here, and this is stated plainly rather than papered over with
an approximate or assumed figure.** Every fetch attempt for De Bernardi et al. (the MDPI landing page,
the MDPI full-text HTML variant, and a ProQuest PDF mirror) returned either HTTP 403 or no extractable
body text (Ch 6.2); the citation itself (authors, year, title, venue, DOI) is confirmed via two
independent sources, but the dataset, architecture configuration, and every reported metric are not. This
project's own Autoencoder results are genuinely poor on both datasets — below the majority baseline on
Dataset B, below-chance ROC-AUC on Dataset A — and echo Isolation Forest's diagnosis almost exactly:
**two structurally unrelated unsupervised architectures (density-based isolation, reconstruction error)
both invert on Dataset A specifically**, strong triangulated evidence (Ch 8.1) that Dataset A's benign
and attack traffic are not separated by the kind of "the minority class looks statistically unusual"
assumption either unsupervised method depends on. Whether De Bernardi et al.'s rule-based, explainable
autoencoder variant would fare differently on this project's data cannot be assessed without a working
comparison number — flagged as an open gap rather than assumed away in either direction.

## 8.3.5 Summary — what actually explains each gap

| model | vs. benchmark | direction | primary cause |
|---|---|---|---|
| XGBoost | Abrahim et al. (2026) | **matches/exceeds** | not primarily framing (plan's prediction didn't hold) — likely standalone-vs-ensemble comparison working in our favour |
| CNN | Li et al. (2024) | recall higher, F1 lower | structural: their model has sequence-level (Transformer) access this project's feature-only schema deliberately excludes (D1) |
| Isolation Forest | Wang et al. (2022) | far below | dataset-specific: attack is not a minority density on either of this project's datasets (Step 2E), independent of benchmark uncertainty |
| Autoencoder | De Bernardi et al. (2025) | not comparable | benchmark paper's numbers are unverifiable, not a methodology gap on this project's side |

**The plan's anticipated single explanation ("our hard framing looks worse because published work uses
the easier framing") turned out to explain none of the four gaps observed.** The real causes are more
specific and more informative: an ensemble-vs-standalone comparison, a feature-access asymmetry between
architectures, a dataset-specific structural property already independently diagnosed in Chapters 5 and
8.1, and one benchmark that simply could not be verified. Reporting the specific cause per model, rather
than one assumed mechanism applied uniformly, is the more defensible use of this comparison — and is
itself consistent with this report's D4 dual-framing design, which exists precisely to let a claim like
"our number looks different because of framing" be checked rather than asserted.
