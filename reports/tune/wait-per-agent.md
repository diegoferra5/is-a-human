# Wait-per-agent-turn — before / after

Generated from live hybrid VAD. Feature `caller_wait_per_agent_turn` is the median of
(positive floor-switch wait) / (preceding agent turn duration).

## Separation

| Split | `caller_wait_per_agent_turn` d | `pos_median` d | corr(ratio, wait) | human mean | synth mean |
| --- | --- | --- | --- | --- | --- |
| train | 0.4159 | 1.8972 | 0.3379 | 0.6448 | 1.4236 |
| val | 0.5383 | 2.363 | 0.2114 | 0.5395 | 1.4611 |

## Tandem (full predict: stacked + disagree override + quiet-talkative)

| | val acc | val AUC | val miss | train acc | behavioural val acc |
| --- | --- | --- | --- | --- | --- |
| before (3 feats) | 0.9718 | 0.9976 | 2 | 0.9645 | 0.9296 |
| after (+ caller_wait_per_agent_turn) | 0.9718 | 0.9984 | 2 | 0.9574 | 0.9296 |

Before features: `caller_response_latency_pos_median_s, agent_talk_ratio, agent_aligned_recovery_cv`  
After features: `caller_response_latency_pos_median_s, agent_talk_ratio, agent_aligned_recovery_cv, caller_wait_per_agent_turn`

## Watched calls (val)

### Before
- `call_569ffb0869eb` human fused=0.0205 ac=0.0205 beh=0.947 wait=2.5 ratio=1.135
- `call_6971b2685c1d` human fused=0.8335 ac=0.8595 beh=0.5749 wait=1.745 ratio=0.4008
- `call_d4518fcb700a` synthetic fused=0.3315 ac=0.8695 beh=0.2054 wait=1.74 ratio=0.6676
- `call_e2d538297f0c` human fused=0.2773 ac=0.8792 beh=0.4586 wait=1.51 ratio=0.3979

### After
- `call_569ffb0869eb` human fused=0.0205 ac=0.0205 beh=0.9464 wait=2.5 ratio=1.135
- `call_6971b2685c1d` human fused=0.8267 ac=0.8595 beh=0.5649 wait=1.745 ratio=0.4008
- `call_d4518fcb700a` synthetic fused=0.3486 ac=0.8695 beh=0.2084 wait=1.74 ratio=0.6676
- `call_e2d538297f0c` human fused=0.2781 ac=0.8792 beh=0.453 wait=1.51 ratio=0.3979

## Val misses

### Before
- `call_6971b2685c1d` human fused=0.8335 ac=0.8595 beh=0.5749 wait=1.745 ratio=0.4008
- `call_d4518fcb700a` synthetic fused=0.3315 ac=0.8695 beh=0.2054 wait=1.74 ratio=0.6676

### After
- `call_6971b2685c1d` human fused=0.8267 ac=0.8595 beh=0.5649 wait=1.745 ratio=0.4008
- `call_d4518fcb700a` synthetic fused=0.3486 ac=0.8695 beh=0.2084 wait=1.74 ratio=0.6676
