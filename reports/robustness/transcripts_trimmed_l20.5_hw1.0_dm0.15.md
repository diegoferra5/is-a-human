# Semantic robustness — transcripts=transcripts_trimmed, l2_gate=0.5, hard_weight=1.0, disagree_margin=0.15

Train OOF: 5-fold × seeds [0, 1, 2], n=282. Gate margin 0.3: 13.1% of calls gated by unsure, 16.2% by unsure-or-disagree; gate combiner fit on ~30 inner rows.

## Train OOF accuracy (mean over seeds)

| scenario | fast | gated_fusion3_all | gated_logit_avg | gated_fit | dgate_logit_avg | dgate_3way_avg | dgate_fusion3_all | always_fusion3_all | semantic_only |
|---|---|---|---|---|---|---|---|---|---|
| clean | 95.7 | 97.6 | 97.8 | 98.2 | 97.5 | 96.5 | 97.6 | 97.3 | 89.2 |
| acoustic_masked | 91.7 | 96.9 | 95.7 | 94.8 | 95.7 | 96.9 | 96.9 | 96.9 | 89.2 |
| behavioral_masked | 86.9 | 91.3 | 91.0 | 91.4 | 91.0 | 92.2 | 91.3 | 91.3 | 89.2 |
| both_masked | 59.9 | 87.9 | 87.8 | 88.4 | 87.8 | 89.2 | 87.9 | 87.9 | 89.2 |
| acoustic_noisy | 94.0 | 96.9 | 96.8 | 96.6 | 96.3 | 94.9 | 96.9 | 96.9 | 89.2 |
| behavioral_noisy | 91.8 | 94.4 | 94.8 | 94.4 | 95.2 | 94.7 | 94.4 | 94.6 | 89.2 |
| both_noisy | 90.2 | 94.1 | 94.6 | 93.1 | 94.3 | 93.7 | 94.1 | 94.6 | 89.2 |
| acoustic_flipped | 76.0 | 88.1 | 86.4 | 84.0 | 86.2 | 79.9 | 88.3 | 92.0 | 89.2 |
| behavioral_flipped | 26.1 | 45.3 | 50.6 | 51.2 | 68.0 | 66.1 | 46.0 | 46.2 | 89.2 |
| semantic_masked | 95.7 | 95.2 | 95.7 | 94.9 | 95.7 | 94.2 | 95.2 | 95.2 | 59.9 |

## Train OOF AUC

| scenario | fast | gated_fusion3_all | gated_logit_avg | gated_fit | dgate_logit_avg | dgate_3way_avg | dgate_fusion3_all | always_fusion3_all | semantic_only |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.993 | 0.996 | 0.995 | 0.995 | 0.995 | 0.995 | 0.996 | 0.998 | 0.946 |
| acoustic_masked | 0.972 | 0.996 | 0.996 | 0.987 | 0.996 | 0.990 | 0.996 | 0.996 | 0.946 |
| behavioral_masked | 0.914 | 0.965 | 0.967 | 0.963 | 0.967 | 0.967 | 0.965 | 0.965 | 0.946 |
| both_masked | 0.434 | 0.940 | 0.945 | 0.946 | 0.945 | 0.946 | 0.940 | 0.940 | 0.946 |
| acoustic_noisy | 0.985 | 0.992 | 0.990 | 0.989 | 0.989 | 0.988 | 0.992 | 0.997 | 0.946 |
| behavioral_noisy | 0.969 | 0.977 | 0.976 | 0.976 | 0.979 | 0.978 | 0.978 | 0.989 | 0.946 |
| both_noisy | 0.963 | 0.975 | 0.974 | 0.972 | 0.974 | 0.973 | 0.977 | 0.989 | 0.946 |
| acoustic_flipped | 0.795 | 0.912 | 0.885 | 0.882 | 0.889 | 0.824 | 0.924 | 0.978 | 0.946 |
| behavioral_flipped | 0.207 | 0.378 | 0.460 | 0.490 | 0.641 | 0.621 | 0.513 | 0.565 | 0.946 |
| semantic_masked | 0.993 | 0.992 | 0.993 | 0.992 | 0.992 | 0.991 | 0.992 | 0.991 | 0.500 |

## Val (71 unseen voices) — scored once. Gated: unsure 8.5%, unsure-or-disagree 15.5%

| scenario | fast acc / auc | gated_fusion3_all acc / auc | gated_logit_avg acc / auc | gated_fit acc / auc | dgate_logit_avg acc / auc | dgate_3way_avg acc / auc | dgate_fusion3_all acc / auc | always_fusion3_all acc / auc | semantic_only acc / auc |
|---|---|---|---|---|---|---|---|---|---|
| clean | 94.4 / 0.996 | 94.4 / 0.998 | 94.4 / 0.999 | 94.4 / 1.000 | 94.4 / 0.999 | 93.0 / 0.999 | 94.4 / 0.998 | 94.4 / 0.998 | 87.3 / 0.952 |
| acoustic_masked | 93.0 / 0.979 | 95.8 / 0.994 | 93.0 / 0.996 | 93.0 / 0.986 | 93.0 / 0.996 | 93.0 / 0.994 | 95.8 / 0.994 | 95.8 / 0.994 | 87.3 / 0.952 |
| behavioral_masked | 85.9 / 0.963 | 87.3 / 0.970 | 88.7 / 0.971 | 87.3 / 0.970 | 88.7 / 0.971 | 87.3 / 0.975 | 87.3 / 0.970 | 87.3 / 0.970 | 87.3 / 0.952 |
| both_masked | 47.9 / 0.500 | 84.5 / 0.952 | 84.5 / 0.952 | 87.3 / 0.952 | 84.5 / 0.952 | 87.3 / 0.952 | 84.5 / 0.952 | 84.5 / 0.952 | 87.3 / 0.952 |
| acoustic_noisy | 94.4 / 0.989 | 94.4 / 0.993 | 94.4 / 0.994 | 93.0 / 0.993 | 94.4 / 0.994 | 90.1 / 0.990 | 94.4 / 0.993 | 95.8 / 0.994 | 87.3 / 0.952 |
| behavioral_noisy | 88.7 / 0.979 | 95.8 / 0.990 | 97.2 / 0.991 | 97.2 / 0.991 | 95.8 / 0.990 | 94.4 / 0.990 | 95.8 / 0.990 | 94.4 / 0.990 | 87.3 / 0.952 |
| both_noisy | 93.0 / 0.984 | 94.4 / 0.985 | 94.4 / 0.986 | 95.8 / 0.986 | 93.0 / 0.990 | 94.4 / 0.990 | 94.4 / 0.990 | 94.4 / 0.990 | 87.3 / 0.952 |
| acoustic_flipped | 85.9 / 0.859 | 88.7 / 0.918 | 88.7 / 0.901 | 88.7 / 0.896 | 88.7 / 0.896 | 71.8 / 0.824 | 88.7 / 0.920 | 91.5 / 0.973 | 87.3 / 0.952 |
| behavioral_flipped | 22.5 / 0.141 | 45.1 / 0.261 | 46.5 / 0.366 | 49.3 / 0.393 | 67.6 / 0.645 | 64.8 / 0.621 | 45.1 / 0.444 | 45.1 / 0.468 | 87.3 / 0.952 |
| semantic_masked | 94.4 / 0.996 | 94.4 / 0.996 | 94.4 / 0.996 | 95.8 / 0.996 | 94.4 / 0.996 | 93.0 / 0.998 | 94.4 / 0.995 | 94.4 / 0.995 | 47.9 / 0.500 |
