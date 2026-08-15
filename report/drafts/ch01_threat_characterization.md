# Chapter 1 — Threat Characterization

## 1.0 Not a DGA project

DGA = rendezvous problem (malware computes candidate C2 domains from a shared seed, TA0011 C2): "does this
domain look machine-generated?" DNS exfiltration = carrier-capacity problem (fixed destination; the domain
name itself is the transport, stolen bytes encoded as subdomain labels, TA0010 Exfiltration): "is data
smuggled inside this query name?" Different mechanism, feature space, failure mode. This project targets
exfiltration only.

## 1.1 MITRE ATT&CK mapping

| ID | Technique | Tactic | Role |
|---|---|---|---|
| **T1048.003** | Exfiltration Over Unencrypted/Obfuscated Non-C2 Protocol | TA0010 | **Primary** |
| T1071.004 | Application Layer Protocol: DNS | TA0011 | The carrier |
| T1132.001 | Data Encoding: Standard Encoding | TA0011 | Payload base64URL-encoded (F2) |
| T1572 | Protocol Tunneling | TA0011 | Broader tunneling family (no sub-technique as of Aug 2026) |

## 1.2 Attack mechanics

1. **Encode** — base64URL, survives DNS hostname character restrictions.
2. **Chunk** — label capped at 63 bytes, FQDN at 253, so payload splits into subdomain labels.
3. **Emit** — ordinary-looking outbound query; why DNS survives egress policies.
4. **Forward** — resolver walks the FQDN to the attacker's nameserver, which logs it; the query *is* the
   payload.

## 1.3 Feature rationale and telemetry mapping

| Family | Security meaning | Raw field(s) | Derived |
|---|---|---|---|
| **F1** Payload volume | More bytes than a lookup needs | A: `len`, `subdomain_length`, `FQDN_count` · B: `PacketLengthMean/Median`, `FlowBytesSent` | `vol_*` |
| **F2** Encoding randomness | Base64 near-uniform char distribution | A: `entropy`, char-class ratios · B: `PacketLengthCoefficientofVariation`, `log1p(PacketLengthVariance)` | `rand_*` |
| **F3** Structural complexity | Segmented/deep names from chunking | A: `labels`, `labels_max` · B: `PacketLengthMode/StdDev` | `struct_*` |
| **F4** Temporal rhythm *(B only)* | Machine-paced, low-variance emission | B: `PacketTimeMean/StdDev/SkewFromMedian` | `time_*` |
| **F5** Endpoint dispersion *(B only)* | Concentrates on few endpoints | B: `FlowSentRate` | `disp_uniqueness` |

B's F2/F3 are statistical proxies (packet-length variability), not text-level equivalents — DoH hides the
label. F4/F5 unobservable on A by construction (stateless schema, no timing/fan-out telemetry) — central
empirical claim (Ch. 5): feature families survive the plaintext→encrypted boundary by different amounts.
Provenance: `schema/unified.py`.

## 1.4 Throughput/stealth tradeoff → heavy vs. light

Max-throughput exfiltration finishes fast but spikes volume; throttled exfiltration looks closer to normal
at the cost of time. Dataset A operationalizes this as **heavy_attack** (323,698) vs. **light_attack**
(53,978) — Nadler et al.'s (2019) "low-throughput exfiltration" (Ch. 2). Predicted stealth advantage does
**not** show up as a detection gap (Ch. 8.1): recall is statistically indistinguishable between subclasses
(99.95% light vs. 99.94% heavy) — the real cost of this threat model is false positives, not a
light-class blind spot.
