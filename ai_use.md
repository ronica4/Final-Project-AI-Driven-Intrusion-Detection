# genai disclosure

this document explains how we used generative AI for the DNS exfiltration detection final project.

## 1. tools used

we used Claude Code (Anthropic) as a coding tool, one instance per teammate working on our own machines
against a shared git repo. full, unedited session logs are included at
`ai_logs/claude_code_teammate_A.json` and `ai_logs/claude_code_teammate_B.json`, per the assignment's
disclosure requirement.

## 2. working process and what we learned

we designed the project ourselves first: a shared feature schema mapping both datasets to comparable
behavioural families, the phase breakdown (ingestion, EDA, four models, ensemble/forensics, report), and
the specific experiments we wanted evidence for, before bringing any of it to the AI. we used the tool
mainly for drafting boilerplate code to our spec, explaining unfamiliar library behaviour, and debugging
once something broke — not for deciding what to build or what conclusion to draw. every real number in
this report came from running our own code against the real datasets and checking it against our own
test suite; we did not take the AI's stated results at face value.

a few concrete things we caught by verifying rather than assuming a first answer was right:

- **SMOTE vs. class weighting.** we had planned to SMOTE every model. once we looked closer, we
  realised resampling and cost-sensitive reweighting are two different answers to the same imbalance
  problem, and picked one per model with a stated reason (XGBoost's built-in `scale_pos_weight`; SMOTE for
  the CNN, which has no such knob; neither for Isolation Forest, which structurally cannot accept
  resampling, or for the Autoencoder, which trains on benign rows only).
- **Leakage, twice.** we deliberately included an obviously-leaky column in two different feature-ranking
  runs (`sld` on the plaintext dataset, `SourceIP` on the encrypted one) to test whether our own pipeline
  would catch it. both times the leaky column dominated feature importance but barely moved the actual F1
  score once we checked — which taught us importance and impact are not the same thing, and that both
  need checking, not just one.
- **Cross-dataset transfer.** we expected a model trained on one dataset to transfer imperfectly to the
  other. the real result was a full collapse to a fixed-guess classifier in almost every case, which we
  traced with a Kolmogorov-Smirnov distribution-shift check rather than just reporting the negative
  result and moving on — we found that even our "payload volume" feature, which we had assumed meant the
  same thing on both sides of the encryption boundary, differs by three orders of magnitude in raw scale
  between datasets.
- **Honest negative results.** several real experiments came back worse than expected (an ensemble
  cascade that underperformed a single model, an autoencoder that scored below chance on one dataset). we
  diagnosed why rather than tuning the number until it looked better, and reported the results as
  measured.

## 3. summary of prompts and questions

**schema and ingestion design.** prompts: draft loader code that projects both a plaintext DNS dataset and
an encrypted DoH dataset into the single feature schema we had already designed, so every downstream
script and evaluation function could stay agnostic to which dataset it was looking at. questions: which
raw columns in each dataset's real header actually match up to the behaviour we wanted (does "payload
volume" mean the same thing when one dataset counts subdomains and the other counts bytes?); how to keep a
dataset's obviously-identifying columns (a raw hostname string, a fixed testbed IP) out of the model's
actual features while still letting us demonstrate what happens if they're left in.

**EDA and feature ranking.** prompts: implement the significance test and effect-size calculation we
wanted for comparing attack vs. benign feature distributions, and a gain-based feature importance ranking
plus a multicollinearity check. questions: why raw p-values are close to meaningless at our dataset's row
counts and what to report instead; why a feature can dominate a model's importance ranking while barely
affecting its score, and how to tell those two things apart with a real before/after experiment rather
than trusting importance alone.

**models.** prompts: implement four model architectures (gradient-boosted trees, an isolation-forest
anomaly detector, a 1D convolutional network, and an autoencoder) against our shared feature set, with
hyperparameters read from one config file rather than hardcoded. questions: why an anomaly detector's core
assumption (the attack is a rare, unusual minority) might not hold on a dataset built for supervised
training with a large attack proportion, and how to prove that with a number (below-chance ROC-AUC) rather
than just observing a bad F1 score.

**cross-dataset transfer and the ensemble cascade.** prompts: implement the train-on-one/test-on-the-other
experiment we designed, and a three-stage cascade that filters cheap-and-obvious traffic before it reaches
a more expensive model. questions: how to tell whether a transfer failure is caused by the two
environments genuinely differing versus a model having memorised its training set, with evidence rather
than a guess; why a cascade's overall recall can never exceed its first stage's recall, no matter how good
the later stages are.

**error forensics and the report.** prompts: pull the actual misclassified rows behind an aggregate
metric, and compute which rows two different models get wrong in common. questions: how to tell "these
two models fail on related rows" apart from "these two models both fail on a lot of rows, so some overlap
is inevitable" — which led us to compute an expected-overlap-under-independence baseline for every model
pair ourselves, rather than trusting a raw overlap percentage on its own.
