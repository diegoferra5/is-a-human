# Tandem head tune — before / after

Generated 2026-09-13 02:19 UTC.

Revert the shipped detector with:

```
cp models/tandem-before-tune.json models/tandem.json
```

## Before (shipped)

- fusion `stacked` · VAD `hybrid`
- stacked val acc **0.9014** AUC 0.9785
- acoustic val acc 0.9014 · features `caller_rms_mean, caller_spectral_flatness_std, caller_spectral_centroid_std, caller_rms_cv, caller_zcr_std`
- behavioural val acc 0.7887 · features `caller_response_latency_cv, agent_aligned_recovery_cv, silence_ratio, agent_aligned_recovery_mean_s, caller_backchannel_count`

## Ablations

| id | VAD | rms_mean | stacked acc | stacked AUC | acoustic acc | behavioural acc |
| --- | --- | --- | --- | --- | --- | --- |
| `hybrid-no-rms-mean` | hybrid | False | 0.9437 | 0.996 | 0.8732 | 0.9296 |
| `hybrid-with-rms-mean` | hybrid | True | 0.9437 | 0.9976 | 0.9014 | 0.9296 |
| `silero-no-rms-mean` | silero | False | 0.9437 | 0.9841 | 0.7606 | 0.9296 |
| `silero-with-rms-mean` | silero | True | 0.9577 | 0.996 | 0.8732 | 0.9296 |

## After (chosen)

- **hybrid-no-rms-mean**: dropped caller_rms_mean (stacked acc hold ≤ 3%); hybrid matched or beat Silero on behavioural/stacked acc.
- fusion `stacked` · VAD `hybrid`
- stacked val acc **0.9437** AUC 0.996
- acoustic `caller_rms_cv, caller_zcr_std, caller_crest_factor_cv, caller_spectral_flatness_std, caller_spectral_centroid_std`
- behavioural `caller_response_latency_pos_median_s, agent_talk_ratio, agent_aligned_recovery_cv`
- mixer weights acoustic=1.6454 behavioural=2.4137
