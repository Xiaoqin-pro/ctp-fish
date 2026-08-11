# CXT-Fish submission v4 revision log

This revision responds to the final reviewer-consensus checklist without adding experiments or changing frozen results.

1. **Related work:** added Huang, Boom, and Fisher (2015) and explicitly positioned trajectory-aware separation and voting as prior work; removed any implied first-use claim for group-disjoint Fish4Knowledge evaluation.
2. **Figure 6:** corrected panel (b) so `+0.62 pp` is labelled `same-class composite`; group-balanced accuracy remains `+0.09 pp` in the tabulated results.
3. **Method definition:** defined foreground sufficiency operationally as label supervision on the context-suppressed view, not as a formal proof of subject-only information sufficiency.
4. **Reproducibility detail:** expanded foreground/background diagnostic-view definitions, ordinary augmentation, donor resize/alignment, and fold-local inner-development construction.
5. **Bootstrap conditioning:** specified recipient recorded group as the cluster unit and fixed donor identities/reuse through the complete seed-3407 recipient-donor manifest; added the frozen fold, seed, checkpoint, and finite-stratum conditioning set.
6. **Table 8:** stated that differences are computed from unrounded cell-level values and retained the one-seed architecture-sensitivity boundary.
7. **Data availability:** clarified the public code/manifest/summary boundary and the need for authorized local Fish4Knowledge data for complete training or per-image replay; trained checkpoints and full per-image outputs are not redistributed.
8. **Final audit:** added `reports/cxt_fish_submission_v4_final_audit.md`, including v4 hashes, rendering checks, evidence-role checks, and the 88-test result.
9. **Terminology alignment:** changed the Table 2 diagnostic question to fine-context suppression and changed the §3.5 phrase to `separately label-predictive`, avoiding a subject-only or independence interpretation.

No model, checkpoint, prediction, metric, split, donor assignment, or official TEST access was changed in v4.
