# CXT-Fish submission v5 consistency audit

## Target

- `paper/CXT-Fish_IMTS_Submission_Ready_v5.docx`
- SHA-256: `A7B8C76BCFC7238E50AE0E0F62E8E85F9E4D4AF06A512D2FB41D500C63781930`

## Structural checks

- Paragraphs: 183
- Tables: 9
- Embedded figures: 8
- Rendered pages: 18
- Corresponding author email: `xiao.qin@sdust.edu.cn`
- Replacement-character or identified mojibake code points: 0

## Core numerical cross-check

The following frozen values appear consistently in the Abstract, Methods/Results discussion, tables, figures, or concluding synthesis as appropriate:

- main ResNet18 cross-class composite effect: `+7.24 pp`;
- conditional paired group-cluster bootstrap interval: `+6.13 to +8.20 pp`;
- ResNet18 clean effect: `-0.21 pp`;
- ordinary-RGB two-view control contrast: `+6.83 pp`;
- donor-subject-suppressed sensitivity: `+10.48 pp`;
- recorded-group-balanced sensitivity: `+4.47 pp`;
- species-group-balanced sensitivity: `+6.73 pp`;
- main clean means: `0.9577` and `0.9556`;
- main cross-class composite means: `0.5189` and `0.5913`;
- main DAR-flip means: `0.2146` and `0.1792`.

## Evidence-tier terminology

- Main evidence is described as a `same-corpus frozen group-disjoint re-evaluation`.
- The focal uncertainty summary is described as a `conditional paired group-cluster bootstrap interval`.
- Donor-realization, F0-2RGB, donor-subject-suppressed, group-weighted, MobileNet, and Route C analyses remain explicitly post-hoc, descriptive, exploratory, architecture-sensitivity, or mechanism-boundary evidence as applicable.
- Occurrences of `independent validation`, `context invariance`, and `backbone-agnostic` appear only in explicit denials or limitations.
- No claim of clean-accuracy improvement, natural-background invariance, external-domain generalization, universal tail improvement, or end-to-end deployment validation is introduced.

## Data and deployment boundaries

- The official Fish4Knowledge TEST partition remains recorded as not accessed.
- The manuscript states that masks are privileged training information and carry annotation cost.
- Inference is restricted to the recognition stage after fish crops are available and uses ordinary RGB with one forward pass.
- Camera, session, date, site, deployment, cross-corpus, detection, tracking, and raw-video pipeline generalization are not claimed.

## Result

PASS. The v5 manuscript is numerically and terminologically aligned with the frozen evidence package. This audit did not recompute or modify experimental results.
