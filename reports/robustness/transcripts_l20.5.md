# Semantic robustness — transcripts=transcripts, l2_gate=0.5

Train OOF: 5-fold × seeds [0, 1, 2], n=282. Gate margin 0.3: 13.1% of calls gated; gate combiner fit on ~30 inner rows.

## Train OOF accuracy (mean over seeds)

| scenario | fast | gated_fusion3_all | gated_logit_avg | gated_fit | always_fusion3_all | semantic_only |
|---|---|---|---|---|---|---|
| clean | 95.7 | 97.4 | 97.6 | 97.9 | 96.8 | 85.6 |
| acoustic_masked | 91.7 | 96.0 | 95.7 | 93.7 | 96.0 | 85.6 |
| behavioral_masked | 86.9 | 90.0 | 89.6 | 88.8 | 90.0 | 85.6 |
| both_masked | 59.9 | 84.2 | 84.0 | 84.9 | 84.2 | 85.6 |
| acoustic_noisy | 94.0 | 96.0 | 95.6 | 95.9 | 96.2 | 85.6 |
| behavioral_noisy | 91.8 | 94.3 | 94.4 | 94.4 | 94.2 | 85.6 |
| both_noisy | 90.2 | 92.7 | 92.9 | 92.1 | 93.0 | 85.6 |
| acoustic_flipped | 76.0 | 85.9 | 84.3 | 83.1 | 87.5 | 85.6 |
| behavioral_flipped | 26.1 | 44.2 | 50.5 | 50.1 | 44.7 | 85.6 |
| semantic_masked | 95.7 | 96.0 | 95.7 | 95.5 | 96.0 | 59.9 |

## Train OOF AUC

| scenario | fast | gated_fusion3_all | gated_logit_avg | gated_fit | always_fusion3_all | semantic_only |
|---|---|---|---|---|---|---|
| clean | 0.993 | 0.996 | 0.995 | 0.995 | 0.995 | 0.928 |
| acoustic_masked | 0.972 | 0.991 | 0.990 | 0.985 | 0.991 | 0.928 |
| behavioral_masked | 0.914 | 0.961 | 0.964 | 0.959 | 0.961 | 0.928 |
| both_masked | 0.434 | 0.920 | 0.926 | 0.927 | 0.920 | 0.928 |
| acoustic_noisy | 0.985 | 0.990 | 0.989 | 0.988 | 0.993 | 0.928 |
| behavioral_noisy | 0.969 | 0.975 | 0.975 | 0.974 | 0.984 | 0.928 |
| both_noisy | 0.963 | 0.972 | 0.972 | 0.971 | 0.983 | 0.928 |
| acoustic_flipped | 0.795 | 0.888 | 0.878 | 0.871 | 0.946 | 0.928 |
| behavioral_flipped | 0.207 | 0.373 | 0.450 | 0.449 | 0.548 | 0.928 |
| semantic_masked | 0.993 | 0.992 | 0.993 | 0.993 | 0.992 | 0.500 |
