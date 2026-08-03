# Phase 1B v1.0 to v1.1 protocol change

This change was made before any Phase 1B model implementation or training.

| Topic | v1.0 | v1.1 |
|---|---|---|
| Batch | P=6, Q=2, K=2 | P=4, Q=3, K=2 |
| Projection head | 512->512->128 | 512->256->128 |
| Temperature | 0.07 | 0.1 |
| C0 | ordinary SupCon | structured-batch CE control, contrastive weight 0 |
| C1 | cross-track SupCon | ordinary SupCon |
| C2 | cross-track plus context losses | cross-track SupCon only |
| Context mechanisms | included in C2 | deferred to Phase 1C |
| Pilot seed | unspecified | configured seed 3407 |

The purpose is to isolate the cross-trajectory positive definition from
structured batching and from context-control mechanisms. No data partition,
checkpoint, validation metric, or locked evaluation boundary changed.
