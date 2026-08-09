# CXT-Fish IMTS manuscript revision log

Revision target: `CXT-Fish_IMTS_Final_ConstructValidity_Manuscript_revised.docx`

This revision uses the frozen reviewer-control/construct-validity snapshot at commit `11624e0f804b8f69dbd878b04a9f430bab03c055`. No model was trained, no checkpoint was changed, no new inference was run, and the official Fish4Knowledge TEST partition remained locked.

## Reviewer issue → manuscript action

| Issue | Action in the revised manuscript |
|---|---|
| Same-corpus outer folds could be misread as independent confirmation | Replaced `post-development`, `confirmation`, and equivalent language with `same-corpus frozen group-disjoint re-evaluation`; explicitly state that development and outer folds partition the same corpus. |
| Main robustness metric was frame-weighted within species | Defined macro-F1 as class-balanced but image-weighted within species; added post-hoc recorded-group-balanced and species–group-balanced sensitivity, without replacing the main estimand. |
| Conditional nature of bootstrap uncertainty | Recast the main interval as a conditional paired group-cluster bootstrap interval, conditional on completed development, frozen models/seeds, and frozen seed-3407 donor realization. |
| Finite cluster counts were not visible | Added the 16 species × 3 outer-fold recorded-group table to Supplementary Table S2 and described the smallest strata as finite-sample limitations. |
| ± notation was underspecified | Table 4/5 captions now state that ± is sample SD across the relevant equally weighted cells/seeds; paired differences use unrounded cell-level values. |
| Contribution wording was too broad | Changed the method contribution to “instantiate a simple label-level foreground-sufficiency objective”; removed any first-use/new-paradigm implication. |
| Related Work needed closest prior art | Added privileged-information references (Vapnik & Vashist 2009; Lopez-Paz et al. 2016), clarified foreground/context recomposition and background-robustness precedents, and positioned CLIB as the closest domain-specific comparator. |
| Training-mask and deployment scope | Added privileged-annotation cost, mask-quality limitation, and recognition-stage-only deployment boundary; explicitly excluded detector/tracker/raw-video claims. |
| Higher acquisition units were not verifiable | Added the limitation that camera/session/deployment/date identities could not be verified from frozen recognition metadata. |
| MobileNet wording could overstate replication | State that foreground improves in 3/3 folds, cross-class composite in 2/3 folds, and that MobileNet is architecture-sensitivity evidence only. |
| Post-hoc construct controls could be overinterpreted | Clearly label donor-realization, F0-2RGB, donor-subject-suppressed, group-weighted, and Route C results as post-hoc or boundary evidence; no control is presented as independent validation. |
| Figures contained stale terminology/numbers | Replaced the embedded scientific figures with reproducible matplotlib outputs based on frozen CSV/JSON artifacts; corrected the two-view control to +6.83 pp and removed confirmation wording from the protocol schematic. |

## New reproducible artifacts

- `experiments/cxt_fish_group_weighted_robustness_per_cell.csv`
- `experiments/cxt_fish_group_weighted_robustness_summary.csv`
- `experiments/cxt_fish_group_weighted_robustness_summary.json`
- `experiments/cxt_fish_outer_recorded_group_counts_16x3.csv`
- `experiments/cxt_fish_outer_recorded_group_counts_16x3_wide.csv`
- `reports/cxt_fish_group_weighted_robustness_sensitivity.md`
- `scripts/analyze_final_group_weighted_sensitivity.py`
- `scripts/plot_cxt_fish_manuscript_figures.py`
- `reports/figures/manuscript_final/fig1_protocol.png` through `fig8_architecture_sensitivity.png`
- `paper/CXT-Fish_IMTS_Supplementary_Material.docx`

