# CXT-Fish Phase 1C F0 context baseline

F0 reuses the frozen A1 seed-3407 checkpoint and is re-evaluated with the Phase 1C validation-only evaluator. It does not retrain A1.

| Validation view | Macro F1 |
|---|---:|
| Original RGB | 0.974727 |
| Foreground with blurred context | 0.879769 |
| Same-class, cross-track context swap | 0.925769 |
| Cross-class context swap | 0.543295 |

The cross-class-swap macro-F1 decrease is 0.431432. Cross-class prediction agreement is 0.734966; DAR is 0.203686 and DAR_flip is 0.201075. These values establish the frozen F0 comparison point only; they are not used to choose foreground construction parameters.

Provenance: frozen A1 checkpoint `outputs/cxt_fish/phase1a_replication/A1_s1_ce_seed3407/best.pt`; Phase 1C manifest SHA-256 `efacaed64e99b25aaa04e8f662d09c9e43a2b8a0a95bf7aa1e190a059e477448`; validation partition only; internal test and outer folds were not accessed.
