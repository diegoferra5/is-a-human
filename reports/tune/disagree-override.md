# Disagreement override — before / after

Rule: if heads disagree and `|p_fused - 0.5| < 0.10`, use the sharper head (`|p - 0.5|` larger). Otherwise keep the stacked mixer.

- before accuracy **0.9437** · misses 4 (agree 1 · mixer 3)
- after accuracy **0.9577** · misses 3 (agree 1 · mixer 2)
- balanced acc 0.9448 → 0.9583

## All 10 disagreements

| call | label | mixer p | after p | ac | beh | mixer ok | after ok | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `call_0e1e2f29bfdc` | human | 0.109 | 0.109 | 0.806 | 0.031 | True | True |  |
| `call_198ef413fe4a` | human | 0.090 | 0.090 | 0.710 | 0.065 | True | True |  |
| `call_569ffb0869eb` | human | 0.551 | 0.021 | 0.021 | 0.947 | False | True | SAVE |
| `call_678ee1dd2242` | human | 0.154 | 0.154 | 0.003 | 0.661 | True | True |  |
| `call_9a3d5b0a64a3` | human | 0.279 | 0.279 | 0.846 | 0.183 | True | True |  |
| `call_bc2c7c34acea` | human | 0.153 | 0.153 | 0.719 | 0.153 | True | True |  |
| `call_bf0384ed27ed` | human | 0.356 | 0.356 | 0.663 | 0.368 | True | True |  |
| `call_d4518fcb700a` | synthetic | 0.332 | 0.332 | 0.870 | 0.205 | False | False |  |
| `call_e2d538297f0c` | human | 0.723 | 0.723 | 0.879 | 0.459 | False | False |  |
| `call_fa26d1186721` | human | 0.127 | 0.127 | 0.677 | 0.149 | True | True |  |

Flipped 1 · saved 1 · broke 0.

Naive max-margin on every disagreement was a wash (save 2 / break 2). This rule only fires when the mixer itself is unsure, which on val is one call.

Revert the rule by setting `disagree_unsure` to `0` in `models/tandem.json` (or load an older code path).

