# Chapter 6.2 — Architecture-Specific Literature Benchmarks

Four papers, one per architecture, applying it to DNS tunneling/exfiltration or DoH detection. Reused in
Ch. 8.3. Two entries carry **"not independently verified"** — ResearchGate/MDPI/ScienceDirect all
returned HTTP 403; the citations themselves are confirmed in every case.

- **Abrahim, H., Hou, W., Zhuang, Y., & Rahman, H. U. (2026).** "Enhancing intrusion detection in
  encrypted DoH traffic through a robust ensemble learning framework." *PLOS ONE*.
  DOI: 10.1371/journal.pone.0345880. Dataset: CIRA-CIC-DoHBrw-2020 (B's family) + DoH-Tunnel-Traffic-HKD.
  XGBoost as meta-learner atop LSTM/GRU. Reported (ensemble-level): Recall=0.9996, Precision=0.9967,
  F1=0.9981.

- **Li, H., Li, Z., Zhang, S., & Pu, X. (2024).** "Malicious DNS detection by combining improved
  transformer and CNN." *Scientific Reports*. DOI: 10.1038/s41598-024-81189-1. Dataset: DGA + MDND, tested
  against CIC-Bell-DNS (A's family). 1D-CNN + improved Transformer ("ITransformer-CNN"). Reported:
  Accuracy=95.67%, Recall=83.21%, F1=88.43%; Precision/FPR not reported.

- **Wang, S., Sun, L., Qin, S., Li, W., & Liu, W. (2022).** "KRTunnel: DNS channel detector for mobile
  devices." *Computers & Security*, 120, 102818. DOI: 10.1016/j.cose.2022.102818. Isolation Forest over
  mobile DNS features. Reported: accuracy 98.1%. **Not independently verified** — from search snippets,
  not the paper itself.

- **De Bernardi, G., Gaggero, G. B., Patrone, F., Zappatore, S., Marchese, M., & Mongelli, M. (2025).**
  "Rule-Based eXplainable Autoencoder for DNS Tunneling Detection." *Computers*, 14(9), 375.
  DOI: 10.3390/computers14090375. Citation confirmed; dataset/config/Recall/FPR/F1 **could not be
  verified** — every fetch returned HTTP 403 or no extractable body. Gap stated plainly, no invented
  numbers.
