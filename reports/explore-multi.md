=== Multi-run exploratory analysis ===

split=train calls=282 human=113 synthetic=169

Top separators (by |effect size|):
  caller_rms_mean: human=0.069±0.041 synthetic=0.150±0.065 d=+1.50 [+1.25, +1.75] (synthetic > human)
  caller_spectral_flatness_std: human=0.040±0.028 synthetic=0.011±0.006 d=-1.43 [-1.64, -1.27] (human > synthetic)
  caller_zcr_std: human=0.070±0.038 synthetic=0.032±0.014 d=-1.37 [-1.61, -1.18] (human > synthetic)
  caller_rms_cv: human=0.617±0.324 synthetic=0.260±0.178 d=-1.37 [-1.63, -1.12] (human > synthetic)
  caller_spectral_centroid_std: human=335.578±222.643 synthetic=111.998±66.328 d=-1.36 [-1.63, -1.17] (human > synthetic)
  agent_aligned_recovery_cv: human=0.992±0.186 synthetic=0.750±0.181 d=-1.32 [-1.55, -1.06] (human > synthetic)
  caller_response_latency_cv: human=0.861±0.125 synthetic=0.693±0.134 d=-1.30 [-1.55, -1.05] (human > synthetic)
  caller_crest_factor_cv: human=0.320±0.128 synthetic=0.184±0.080 d=-1.28 [-1.51, -1.02] (human > synthetic)

split=val calls=71 human=37 synthetic=34

Top separators (by |effect size|):
  caller_rms_mean: human=0.066±0.031 synthetic=0.153±0.045 d=+2.26 [+1.74, +2.97] (synthetic > human)
  caller_crest_factor_cv: human=0.299±0.100 synthetic=0.156±0.049 d=-1.82 [-2.48, -1.52] (human > synthetic)
  caller_zcr_std: human=0.080±0.035 synthetic=0.032±0.012 d=-1.81 [-2.35, -1.51] (human > synthetic)
  caller_rms_cv: human=0.605±0.283 synthetic=0.244±0.163 d=-1.56 [-2.07, -1.17] (human > synthetic)
  caller_crest_factor_std: human=2.061±1.144 synthetic=0.828±0.292 d=-1.48 [-2.23, -1.20] (human > synthetic)
  caller_spectral_centroid_std: human=406.548±262.927 synthetic=128.287±45.422 d=-1.47 [-2.01, -1.21] (human > synthetic)
  caller_spectral_flatness_std: human=0.040±0.027 synthetic=0.012±0.004 d=-1.44 [-1.92, -1.17] (human > synthetic)
  agent_talk_ratio: human=0.556±0.049 synthetic=0.440±0.103 d=-1.44 [-2.12, -1.09] (human > synthetic)

split_rank_correlation=0.818
stable_top_features=caller_rms_mean, caller_zcr_std, caller_rms_cv
bootstrap_runs=200

Logistic baseline (train on train, evaluate on val):
  train accuracy=0.869 f1=0.891 auc=0.922
  val   accuracy=0.845 f1=0.845 auc=0.963
