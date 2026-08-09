# CXT-Fish submission v4 final consistency audit

Date: 2026-08-09  
Audit target: `paper/CXT-Fish_IMTS_Submission_Ready_v4.docx` and its rendered PDF  
Scope: final wording, evidence-tier, figure-label, reference, and artifact-boundary checks only. No training, new inference, or official TEST access was performed.

## Final artifacts

| Artifact | SHA-256 |
|---|---|
| `paper/CXT-Fish_IMTS_Submission_Ready_v4.docx` | `8f6032f29f2e616bac30a02be14a249902858a0c7ca63007c7132726151a3f8e` |
| `paper/CXT-Fish_IMTS_Submission_Ready_v4.pdf` | `4c2b4287c281107a7cf1894680be7fda7fcd92690742ef64c7b6847e27e7a23a` |
| `paper/CXT-Fish_IMTS_Supplementary_Material_v2.docx` | `2fed83ab84dd3533dc5d8e0cea9b3df773ccdfc4f2628735eb7a7ceca9d74a67` |
| `reports/figures/manuscript_submission_v3/fig6_main_results.png` | `f5361b24decc10e745462b70159c52ae88c2aac699ece6a0759f150e36938b0d` |

## Required revision checks

| Check | Result |
|---|---|
| Huang et al. (2015) trajectory-aware prior added and cited | PASS |
| No claim that group-disjoint Fish4Knowledge evaluation is the first trajectory-aware split | PASS |
| Fig. 6(b) label reports `same-class composite` for `+0.62 pp` | PASS |
| Table 2 uses `fine-context suppression` rather than subject-only language | PASS |
| §3.5 uses `separately label-predictive` rather than `independently label-predictive` | PASS |
| Foreground sufficiency defined operationally, without a subject-only proof claim | PASS |
| Foreground-only/background-only, augmentation, donor alignment, and split construction defined | PASS |
| Bootstrap conditions identify recipient recorded group as the cluster unit | PASS |
| Donor identities and reuse are fixed by the complete seed-3407 manifest | PASS |
| Fold allocation, fold × species strata, frozen checkpoints/seeds, and conditioning set stated | PASS |
| Table 8 states that differences use unrounded cell-level values | PASS |
| Data Availability distinguishes public summaries/code from authorized local-data replay | PASS |
| Corresponding author email is `xiao.qin@sdust.edu.cn` | PASS |
| Primary evidence remains same-corpus frozen group-disjoint re-evaluation | PASS |
| Conditional bootstrap interval remains restricted to the focal ResNet18 contrast | PASS |
| MobileNet remains one-seed architecture-sensitivity evidence only | PASS |
| No new training, checkpoint, inference, or official TEST access | PASS |

## Numeric and evidence-boundary checks

The v4 manuscript preserves the frozen headline values and their roles: ResNet18 cross-class donor-context-composite macro-F1 change `+7.24 pp`, conditional paired group-cluster bootstrap interval `+6.13 to +8.20 pp`, mean clean macro-F1 change `−0.21 pp`, DAR-flip lower in `9/9` paired cells, group-weighted sensitivities `+4.47 pp` and `+6.73 pp`, and MobileNet as architecture-sensitivity evidence with heterogeneous cross-class effects. The manuscript continues to state that the study is not independent validation, external validation, natural-background invariance, cross-camera/site generalization, or end-to-end deployment validation.

Macro-F1 is described as class-balanced across species but image-weighted within species. The bootstrap is described as conditional on the completed development process, fixed outer-fold allocation, frozen models/seeds, fixed recipient-donor realization, and finite fold × species group strata. Donors are not treated as a second resampling cluster.

## Rendering check

The v4 PDF renders to 19 pages at US Letter size. Page-level inspection confirms:

- Section 3.10 and Table 3 remain together;
- Table 6 is kept together on its own page rather than leaving a single donor row stranded;
- Table 7 remains on one page;
- Reproducibility, Table 9, and declarations are not separated by a sparse heading-only page;
- references are alphabetized, with Huang et al. in the H position;
- figures and captions are present without clipping or overflow.

## Test status

Repository test suite: `88 passed in 11.87s` using the repository `.venv`.

**Audit conclusion: PASS.** The v4 manuscript is a wording, construct-validity, figure-label, reference, and reproducibility-boundary revision only. Existing experimental values and evidence roles were not changed.
