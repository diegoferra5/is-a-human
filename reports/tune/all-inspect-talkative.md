# Tandem inspection (all)

Generated 2026-09-13 06:15 UTC. fusion `stacked` · VAD `hybrid`.

- calls 353 (synthetic 203 / human 150)
- accuracy **0.966** · balanced acc **0.9635**
- TPR (synthetic) 0.9803 · TNR (human) 0.9467
- misses **12** (agree 5 · mixer picked wrong head 7)
- heads disagree 60 (mixer still right on 53)

## By split

- **train** acc 0.9645 · balanced 0.9602 · misses 10 / 282
- **val** acc 0.9718 · balanced 0.9718 · misses 2 / 71

## Misses

| call | split | label | fused p | acoustic p | behavioural p | kind | pos_median | agent_talk | recovery_cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `call_1761edbd6952` | train | human | 0.611 | 0.000 | 1.000 | miss_mixer | 5.710 | 0.503 | 0.911 |
| `call_24e1989c4d6d` | train | synthetic | 0.325 | 0.541 | 0.432 | miss_mixer | 0.900 | 0.457 | 0.627 |
| `call_4ac92327c83b` | train | human | 0.657 | 0.713 | 0.527 | miss_agree | 1.190 | 0.395 | 0.946 |
| `call_6dfa70c50ea5` | train | synthetic | 0.324 | 0.676 | 0.336 | miss_mixer | 0.100 | 0.306 | 0.804 |
| `call_8834a90f11cd` | train | human | 0.952 | 0.952 | 0.272 | miss_mixer | 1.180 | 0.515 | 0.769 |
| `call_c6daaae3eb0f` | train | human | 0.723 | 0.589 | 0.664 | miss_agree | 1.990 | 0.527 | 0.797 |
| `call_cbad95f3aabc` | train | synthetic | 0.373 | 0.835 | 0.259 | miss_mixer | 1.980 | 0.595 | 0.993 |
| `call_d80d9f6f7884` | train | human | 0.552 | 0.555 | 0.570 | miss_agree | 1.500 | 0.497 | 0.709 |
| `call_e0c1f661d9ca` | train | human | 0.924 | 0.686 | 0.837 | miss_agree | 1.590 | 0.395 | 0.796 |
| `call_e4d9c2e0947a` | train | human | 0.885 | 0.885 | 0.310 | miss_mixer | 1.235 | 0.489 | 0.855 |
| `call_6971b2685c1d` | val | human | 0.834 | 0.860 | 0.575 | miss_agree | 1.745 | 0.528 | 0.741 |
| `call_d4518fcb700a` | val | synthetic | 0.332 | 0.870 | 0.205 | miss_mixer | 1.740 | 0.477 | 1.372 |

Acoustic head on misses:

| call | rms_cv | zcr_std | crest_cv | flatness_std | centroid_std |
| --- | --- | --- | --- | --- | --- |
| `call_1761edbd6952` | 1.153 | 0.107 | 0.498 | 0.096 | 652.9 |
| `call_24e1989c4d6d` | 0.763 | 0.036 | 0.353 | 0.013 | 187.4 |
| `call_4ac92327c83b` | 0.269 | 0.051 | 0.217 | 0.016 | 232.6 |
| `call_6dfa70c50ea5` | 0.542 | 0.041 | 0.175 | 0.020 | 160.7 |
| `call_8834a90f11cd` | 0.112 | 0.017 | 0.223 | 0.005 | 46.5 |
| `call_c6daaae3eb0f` | 0.270 | 0.051 | 0.296 | 0.021 | 150.2 |
| `call_cbad95f3aabc` | 0.130 | 0.050 | 0.110 | 0.017 | 143.3 |
| `call_d80d9f6f7884` | 0.455 | 0.065 | 0.277 | 0.010 | 296.6 |
| `call_e0c1f661d9ca` | 0.341 | 0.027 | 0.221 | 0.030 | 107.8 |
| `call_e4d9c2e0947a` | 0.122 | 0.044 | 0.214 | 0.006 | 190.8 |
| `call_6971b2685c1d` | 0.368 | 0.029 | 0.258 | 0.009 | 91.8 |
| `call_d4518fcb700a` | 0.208 | 0.038 | 0.146 | 0.015 | 107.4 |

## How to read `error_kind`

- `miss_agree` — both heads on the wrong side of 0.5; a new feature is the only fix.
- `miss_mixer` — heads disagree and fusion followed the wrong one; threshold/mixer, not a new layer.
- `ok_disagree` — heads disagree but fusion followed the right one.
