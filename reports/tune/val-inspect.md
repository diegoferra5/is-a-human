# Tandem inspection (val)

Generated 2026-09-13 02:48 UTC. fusion `stacked` · VAD `hybrid`.

- calls 71 (synthetic 34 / human 37)
- accuracy **0.9577** · balanced acc **0.9583**
- TPR (synthetic) 0.9706 · TNR (human) 0.9459
- misses **3** (agree 1 · mixer picked wrong head 2)
- heads disagree 10 (mixer still right on 8)

## By split

- **val** acc 0.9577 · balanced 0.9583 · misses 3 / 71

## Misses

| call | split | label | fused p | acoustic p | behavioural p | kind | pos_median | agent_talk | recovery_cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `call_6971b2685c1d` | val | human | 0.834 | 0.860 | 0.575 | miss_agree | 1.745 | 0.528 | 0.741 |
| `call_d4518fcb700a` | val | synthetic | 0.332 | 0.870 | 0.205 | miss_mixer | 1.740 | 0.477 | 1.372 |
| `call_e2d538297f0c` | val | human | 0.723 | 0.879 | 0.459 | miss_mixer | 1.510 | 0.481 | 0.893 |

Acoustic head on misses:

| call | rms_cv | zcr_std | crest_cv | flatness_std | centroid_std |
| --- | --- | --- | --- | --- | --- |
| `call_6971b2685c1d` | 0.368 | 0.029 | 0.258 | 0.009 | 91.8 |
| `call_d4518fcb700a` | 0.208 | 0.038 | 0.146 | 0.015 | 107.4 |
| `call_e2d538297f0c` | 0.285 | 0.031 | 0.220 | 0.009 | 121.9 |

## How to read `error_kind`

- `miss_agree` — both heads on the wrong side of 0.5; a new feature is the only fix.
- `miss_mixer` — heads disagree and fusion followed the wrong one; threshold/mixer, not a new layer.
- `ok_disagree` — heads disagree but fusion followed the right one.
