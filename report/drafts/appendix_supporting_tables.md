# Appendix D — Supporting Tables

Full versions of tables summarised in the report body, moved here to stay within the 15-page body budget.

## D.1 Cliff's delta, all Dataset A features (referenced in Ch. 3.2)

| feature | Cliff's δ | verdict |
|---|---|---|
| vol_primary | +0.242 | small |
| vol_secondary | +0.551 | large |
| vol_total | +0.564 | large |
| rand_entropy | −0.125 | negligible |
| rand_dispersion | +0.591 | large |
| struct_segments | +0.533 | large |
| struct_max_segment | −0.145 | negligible |

## D.2 Variance Inflation Factor, Dataset A (referenced in Ch. 4.4)

| feature | VIF | flagged (>10) |
|---|---|---|
| vol_primary | 128.3 | yes |
| vol_secondary | 113.6 | yes |
| vol_total | 136.3 | yes |
| rand_entropy | 2.33 | no |
| rand_dispersion | 14.4 | yes |
| struct_segments | 100.5 | yes |
| struct_max_segment | 168.0 | yes |

## D.3 Kolmogorov–Smirnov distribution shift, intersection columns (referenced in Ch. 5.2)

| feature | family | KS | mean (A) | mean (B) | ratio (B/A) |
|---|---|---|---|---|---|
| vol_primary | F1 | 1.0000 | 12.39 | 173.03 | ×14.0 |
| vol_secondary | F1 | 1.0000 | 5.83 | 95.19 | ×16.3 |
| vol_total | F1 | 1.0000 | 21.81 | 40,523.60 | ×1,858.6 |
| struct_segments | F3 | 1.0000 | 4.68 | 70.79 | ×15.1 |
| rand_dispersion | F2 | 0.9897 | 0.89 | 9.34 | ×10.5 |
| struct_max_segment | F3 | 0.9011 | 8.17 | 220.02 | ×26.9 |
| rand_entropy | F2 | 0.8916 | 2.48 | 1.00 | ×0.40 |

## D.4 Cross-model pairwise error overlap vs. chance baseline, Dataset A (referenced in Ch. 8.1.4)

| pair | error | actual | expected (indep.) | ratio |
|---|---|---|---|---|
| XGBoost vs. CNN | FN | 42 | 0.1 | 407× higher |
| XGBoost vs. CNN | FP | 46,775 | 18,954 | 2.47× higher |
| XGBoost vs. IsoForest | FN | 22 | 54 | 2.4× lower |
| XGBoost vs. IsoForest | FP | 3,489 | 14,720 | 4.2× lower |
| XGBoost vs. AE | FN | 40 | 57 | 1.4× lower |
| XGBoost vs. AE | FP | 1,128 | 2,310 | 2.1× lower |
| IsoForest vs. CNN | FN | 15 | 174 | 11.6× lower |
| IsoForest vs. CNN | FP | 3,457 | 14,711 | 4.3× lower |
| IsoForest vs. AE | FN | 97,731 | 95,452 | ≈chance (1.02×) |
| IsoForest vs. AE | FP | 5,327 | 1,793 | 3.0× higher |
| CNN vs. AE | FN | 92 | 184 | 2.0× lower |
| CNN vs. AE | FP | 1,109 | 2,309 | 2.1× lower |
