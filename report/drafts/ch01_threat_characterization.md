# Chapter 1 — Threat Characterization

## 1.0 Why this is not a DGA project

Domain Generation Algorithms (DGA) and DNS exfiltration are both DNS-based threats, but they solve
different adversary problems and sit under different MITRE ATT&CK tactics.

**DGA is a rendezvous problem.** Malware needs to find its command-and-control (C2) server without a
hardcoded, block-listable address, so client and server independently compute a rolling set of
candidate domains from a shared seed (date, secret) and try to resolve them until one succeeds. The
question a DGA detector answers is: *does this domain look machine-generated?* This falls under
**TA0011 (Command and Control)**.

**DNS exfiltration is a carrier-capacity problem.** The attacker already has a foothold and a fixed
destination; the question is how to move stolen bytes out of a network that blocks or inspects every
other outbound channel but almost never blocks outbound DNS. The domain name itself becomes the
transport, encoding attacker data as subdomain labels. The question our detector answers is: *is data
being smuggled inside this query name?* This falls under **TA0010 (Exfiltration)**.

These differ in mechanism (algorithmic generation vs. payload encoding), in feature space (lexical
"does this look like a word" features for DGA vs. volumetric/entropy/structural features for
exfiltration), and in failure mode (a DGA detector that is shown an exfiltration channel is answering
the wrong question entirely, and vice versa). This project targets exfiltration only.

## 1.1 MITRE ATT&CK mapping

| ID | Technique | Tactic | Role in this threat |
|---|---|---|---|
| **T1048.003** | Exfiltration Over Unencrypted/Obfuscated Non-C2 Protocol | TA0010 Exfiltration | **Primary.** DNS is abused as an exfiltration channel outside the malware's normal C2 protocol. |
| T1071.004 | Application Layer Protocol: DNS | TA0011 Command and Control | The carrier protocol — DNS queries are the wire format the data rides on. |
| T1132.001 | Data Encoding: Standard Encoding | TA0011 Command and Control | Payload is base64URL-encoded before being split into query labels (F2 feature family). |
| T1572 | Protocol Tunneling | TA0011 Command and Control | Broader technique family for encapsulating one protocol inside another; DNS tunneling is a concrete instance. |

> **Correction to an earlier plan draft:** the plan text cited "T1572.002." Verified against the live
> ATT&CK Enterprise matrix (August 2026): **T1572 currently has no sub-techniques** — cite it as the
> bare technique, not T1572.002. T1048.003, T1071.004, and T1132.001 were all verified correct as
> written. I've corrected the PROJECT_PLAN.md reference alongside this draft.

## 1.2 Attack mechanics

1. **Encode.** The exfiltrated payload (a file, a credential dump, a command's output) is base64URL-encoded
   so every byte survives DNS's hostname character restrictions.
2. **Chunk.** DNS caps each label at 63 bytes and the full FQDN at 253 bytes, so the encoded payload is
   split into a sequence of labels, each becoming one subdomain level (`<chunk1>.<chunk2>...<attacker-domain>`).
3. **Emit.** The client issues an ordinary-looking DNS query for the constructed FQDN. No inbound
   connection to the victim network is required — the exfiltration is entirely outbound, which is why
   DNS survives in almost every egress policy that blocks everything else.
4. **Forward.** The organization's recursive resolver, doing its normal job, walks the FQDN up to the
   attacker-registered domain and forwards the query to the attacker's authoritative nameserver — which
   simply logs the query. The "response" is irrelevant; the query *is* the payload.

## 1.3 Theoretical feature rationale, by family

The unified feature schema (`schema/unified.py`) groups the project's 11 engineered features into five
behavioural families. Each maps to one step of the attack mechanics above:

| Family | Security meaning | Maps to |
|---|---|---|
| **F1 — Payload volume** | A channel meant to carry only small lookups now carries far more bytes per unit than normal traffic, because the payload itself is the smuggled data. | Step 2 (chunking produces long, high-volume queries) |
| **F2 — Encoding randomness** | Base64URL-encoded binary data has a near-uniform character distribution (high Shannon entropy); legitimate hostnames are drawn from a much smaller, human-readable alphabet and are not statistically random. | Step 1 (encoding) |
| **F3 — Structural complexity** | Chunking to fit the 63-byte label / 253-byte FQDN limits produces unusually deep, unusually segmented names compared to ordinary 2–4 label hostnames. | Step 2 (chunking) |
| **F4 — Temporal rhythm** *(Dataset B only)* | Automated exfiltration issues queries on a machine-paced, low-variance schedule; human-driven browsing does not. | Step 3 (emission cadence) |
| **F5 — Endpoint dispersion** *(Dataset B only)* | Exfiltration channels concentrate traffic toward a small number of attacker-controlled endpoints rather than the broad destination spread of normal browsing. | Step 4 (forwarding target) |

F4 and F5 are unobservable in Dataset A by construction — Dataset A's stateless per-query schema simply
does not carry timing or destination-fan-out telemetry (see the INTERSECTION vs. B_ONLY split below).
This is not a data-cleaning gap to explain away; it is itself the project's central empirical claim
(§1.4 and Chapter 5): different feature families survive the plaintext→encrypted boundary by different
amounts, and F4/F5 are the families that do not survive it at all when only stateless per-query
telemetry is available.

## 1.4 Telemetry & feature mapping

| Behaviour | Telemetry source | Raw field(s) | Derived feature | Explanation |
|---|---|---|---|---|
| Query carries more bytes than a normal lookup | Dataset A: passive DNS query log · Dataset B: DoH flow record | A: `len`, `subdomain_length`, `FQDN_count` · B: `PacketLengthMean`, `PacketLengthMedian`, `FlowBytesSent` | `vol_primary`, `vol_secondary`, `vol_total` (F1) | Direct length/volume counters from each vantage point's native telemetry; A measures the query string itself, B measures the encrypted flow's packet-size statistics. |
| Query label looks statistically random | A: query log · B: DoH flow record | A: `entropy`, `(numeric+special+upper)/len` · B: `PacketLengthCoefficientofVariation`, `log1p(PacketLengthVariance)` | `rand_entropy`, `rand_dispersion` (F2) | A measures character-level entropy directly on the label text. B has no visible label text (DoH encrypts the query) so it substitutes a *statistical proxy* — variability in packet-length, which base64-encoded high-entropy payloads tend to inflate — not a semantic equivalent. |
| Name is unusually segmented/deep | A: query log · B: DoH flow record | A: `labels`, `labels_max` · B: `PacketLengthMode`, `PacketLengthStandardDeviation` | `struct_segments`, `struct_max_segment` (F3) | Same proxy relationship as F2: A counts actual label structure; B infers a weak structural correlate from packet-length shape, since label boundaries are invisible once encrypted. |
| Queries fire on a machine-regular cadence | B only: DoH flow record | B: `PacketTimeMean`, `PacketTimeStandardDeviation`, `PacketTimeSkewFromMedian` | `time_central`, `time_dispersion`, `time_skew` (F4) | Dataset A is a stateless per-query table with no inter-arrival timing captured at all — this family is unobservable on that side by design (Decision D1). |
| Traffic concentrates on few destinations | B only: DoH flow record | B: `FlowSentRate` | `disp_uniqueness` (F5) | Same unobservability as F4 — Dataset A carries no destination-dispersion telemetry. |

*(Full column-level provenance, including every raw field dropped and why, is in `schema/unified.py`'s
`COLUMN_SOURCE` / `DROPPED_COLUMNS` dicts and `docs/header_reconciliation_exf2021.md` /
`docs/header_reconciliation_dohbrw2020.md`.)*

## 1.5 The throughput/stealth tradeoff → heavy vs. light

An attacker who exfiltrates at maximum DNS throughput (large chunks, high query rate) finishes fast but
produces an obvious volume spike — trivially caught by any threshold-based detector. An attacker who
throttles the rate down produces individual queries that look statistically closer to normal traffic,
at the cost of taking far longer to move the same data. This tradeoff is exactly what the
CIC-Bell-DNS-EXF-2021 dataset operationalizes as its **heavy_attack** (323,698 samples) vs.
**light_attack** (53,978 samples) subclasses, and it is the direct throughput-vs-stealth problem
Nadler et al. (2019) first formalized as "low-throughput exfiltration" (Chapter 2). Whether this
predicted stealth advantage actually shows up as a detection gap on our data is an empirical question
answered in Chapter 8.1 — and, notably, it does **not** show up the way the prior literature would
predict (see the Step 3A cross-reference there): recall is statistically indistinguishable between the
two subclasses on Dataset A (99.95% light vs. 99.94% heavy); the real cost of this threat model turns
out to be false positives, not a light-class blind spot.
