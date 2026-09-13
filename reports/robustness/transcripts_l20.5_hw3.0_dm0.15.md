# Semantic robustness — transcripts=transcripts, l2_gate=0.5, hard_weight=3.0, disagree_margin=0.15

Train OOF: 5-fold × seeds [0, 1, 2], n=282. Gate margin 0.3: 13.1% of calls gated by unsure, 16.2% by unsure-or-disagree; gate combiner fit on ~30 inner rows.

## Train OOF accuracy (mean over seeds)

| scenario | fast | gated_fusion3_all | gated_logit_avg | gated_fit | dgate_logit_avg | dgate_3way_avg | dgate_fusion3_all | always_fusion3_all | semantic_only |
|---|---|---|---|---|---|---|---|---|---|
| clean | 95.7 | 97.4 | 97.5 | 98.0 | 97.3 | 96.7 | 97.3 | 96.8 | 85.7 |
| acoustic_masked | 91.7 | 95.9 | 95.3 | 93.1 | 95.3 | 96.0 | 95.9 | 95.9 | 85.7 |
| behavioral_masked | 86.9 | 90.3 | 89.8 | 88.8 | 89.8 | 91.1 | 90.3 | 90.3 | 85.7 |
| both_masked | 59.9 | 84.5 | 84.5 | 85.3 | 84.5 | 85.7 | 84.5 | 84.5 | 85.7 |
| acoustic_noisy | 94.0 | 96.0 | 96.0 | 96.0 | 95.7 | 94.3 | 96.0 | 96.2 | 85.7 |
| behavioral_noisy | 91.8 | 94.4 | 94.4 | 94.1 | 94.6 | 94.1 | 94.3 | 94.4 | 85.7 |
| both_noisy | 90.2 | 92.7 | 92.9 | 92.1 | 93.0 | 92.9 | 92.9 | 93.0 | 85.7 |
| acoustic_flipped | 76.0 | 86.1 | 84.5 | 83.9 | 83.9 | 74.3 | 86.2 | 87.8 | 85.7 |
| behavioral_flipped | 26.1 | 43.9 | 49.8 | 49.6 | 67.4 | 61.7 | 44.6 | 44.4 | 85.7 |
| semantic_masked | 95.7 | 96.0 | 95.7 | 95.5 | 95.7 | 94.2 | 96.0 | 96.0 | 59.9 |

## Train OOF AUC

| scenario | fast | gated_fusion3_all | gated_logit_avg | gated_fit | dgate_logit_avg | dgate_3way_avg | dgate_fusion3_all | always_fusion3_all | semantic_only |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.993 | 0.995 | 0.995 | 0.995 | 0.995 | 0.994 | 0.995 | 0.996 | 0.926 |
| acoustic_masked | 0.972 | 0.991 | 0.990 | 0.983 | 0.990 | 0.984 | 0.991 | 0.991 | 0.926 |
| behavioral_masked | 0.914 | 0.963 | 0.965 | 0.958 | 0.965 | 0.966 | 0.963 | 0.963 | 0.926 |
| both_masked | 0.434 | 0.917 | 0.925 | 0.924 | 0.925 | 0.926 | 0.917 | 0.917 | 0.926 |
| acoustic_noisy | 0.985 | 0.990 | 0.989 | 0.989 | 0.989 | 0.986 | 0.990 | 0.993 | 0.926 |
| behavioral_noisy | 0.969 | 0.975 | 0.975 | 0.975 | 0.977 | 0.976 | 0.976 | 0.985 | 0.926 |
| both_noisy | 0.963 | 0.972 | 0.972 | 0.970 | 0.973 | 0.972 | 0.974 | 0.983 | 0.926 |
| acoustic_flipped | 0.795 | 0.888 | 0.878 | 0.871 | 0.878 | 0.808 | 0.896 | 0.945 | 0.926 |
| behavioral_flipped | 0.207 | 0.370 | 0.448 | 0.448 | 0.616 | 0.595 | 0.505 | 0.542 | 0.926 |
| semantic_masked | 0.993 | 0.992 | 0.993 | 0.993 | 0.992 | 0.991 | 0.992 | 0.992 | 0.500 |
