# CXT-Fish submission v6 consistency audit

## Target artifacts

- `paper/CXT-Fish_IMTS_Submission_Ready_v6.docx`
  - SHA-256: `1CAA9454D8F8BAF548B792B437F33F0461DB1E2E4D1C5337E318EF41BE771C6D`
- `paper/CXT-Fish_IMTS_Submission_Ready_v6.pdf`
  - SHA-256: `3990CCDFF6376F2AB2ADCD9D5FA0F43B101C9EC7D8CAB42696461CFF9397760F`
- `paper/CXT-Fish_IMTS_Supplementary_Material_v2.docx`
  - SHA-256: `E79953A1E8649E7BA79E9C742F2197C76D99074CBF2BEA425DB31FB75A674987`
- `paper/CXT-Fish_IMTS_Supplementary_Material_v2.pdf`
  - SHA-256: `14E0A2D0F7CFDF8C269155301728ECFCE45A56ED8DD459AB29F074EAE9217A4B`

## Structural and visual checks

- Main manuscript: 183 paragraphs, 9 tables, 18 rendered pages, and all eight manuscript figures replaced by repository figure sources.
- Supplementary material: 2 rendered pages; fold-local tier table `Table S2b` is present next to the 16 × 3 recorded-group table.
- Corresponding author email: `xiao.qin@sdust.edu.cn`.
- Main manuscript and supplement were rendered through Word PDF export and visually inspected page by page; no clipping, table overflow, title-only page, or short-table split was observed.

## Core numerical cross-check

The frozen values remain unchanged and are present in the v6 manuscript:

- ResNet18 cross-class composite effect: `+7.24 pp`;
- conditional paired group-cluster bootstrap interval: `+6.13 to +8.20 pp`;
- ResNet18 clean effect: `-0.21 pp`;
- ordinary-RGB two-view control contrast: `+6.83 pp`;
- donor-subject-suppressed sensitivity: `+10.48 pp`;
- recorded-group-balanced sensitivity: `+4.47 pp`;
- species-group-balanced sensitivity: `+6.73 pp`.

## Evidence and reproducibility boundaries

- Main evidence is described as a `same-corpus frozen group-disjoint re-evaluation`.
- The focal uncertainty summary is a `conditional paired group-cluster bootstrap interval`.
- Outer tail tiers are fold-local and derived from each fold's outer-training frequencies; the exact mappings are listed in Supplementary Table S2b.
- Full split manifests remain subject to the repository's ignore/data-access policy; the manuscript now promises deterministic reconstruction and hashes rather than public split-file redistribution.
- The frozen primary and five sensitivity donor realizations are exported without absolute paths in `reports/portable_frozen_donor_manifest.csv`; the resolver is documented in `reports/portable_frozen_donor_manifest_resolver.md`.
- Reviewer-facing package labels were removed from the formal manuscript tables and prose; the text uses `post-hoc validity controls` and `validity-control protocol`.
- Official Fish4Knowledge TEST access remains false. No training, new inference, or numeric result selection was performed for v6.

## Result

PASS. This audit targets the actual v6 submission file and its rendered PDF. It records wording, evidence-tier, tail-tier, and portable-manifest repairs without recomputing or modifying experimental results.
