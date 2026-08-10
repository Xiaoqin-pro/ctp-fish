# CXT-Fish submission v5 figure and prose revision

## Scope

This revision is limited to manuscript writing, figure design, and pagination. It introduces no training, inference, new data access, model selection, numerical change, or post-hoc result deletion.

## Figure redesign

All eight manuscript figures were regenerated in PNG, PDF, and SVG form under `reports/figures/manuscript_submission_v4/`. The revision replaces presentation-like cards and decorative styling with a compact journal grammar:

- data and estimands carry the visual emphasis;
- colour is reserved for semantic distinctions rather than decoration;
- uncertainty, pairing, and evidence roles are encoded directly;
- schematic panels use thin vector geometry and explicit legends;
- typography, panel labels, line widths, and axis formatting are consistent;
- no generative image is used as scientific evidence.

The design pass was informed by the restrained data-first layout used in recent *Intelligent Marine Technology and Systems* articles and by the problem–design–result–boundary narrative common in higher-tier empirical journals. It does not copy any published figure.

## Prose revision

The main prose now follows a tighter sequence:

1. identify the correlation/evaluation problem;
2. define the recorded-group and synthetic-intervention estimands;
3. introduce the minimal training intervention;
4. report the main paired result together with the clean-performance guardrail;
5. separate main frozen evidence from bounded post-hoc controls;
6. state the same-corpus, synthetic-intervention, recognition-stage, and mask-annotation limits at the point of interpretation.

The Abstract, contribution statement, main Results synthesis, Discussion opening, and Conclusion were rewritten accordingly. The manuscript retains the terms `same-corpus frozen group-disjoint re-evaluation` and `conditional paired group-cluster bootstrap interval`, and does not claim independent validation, natural-context invariance, backbone-agnostic performance, or end-to-end deployment validation.

## Pagination repair

- Section 3.10 and Table 3 are kept together.
- Section 4.6 and the complete three-row Table 7 appear together on page 12.
- Section 4.9 and Table 8 begin together on page 14, eliminating an isolated table-header row.
- The final Word export is 18 pages.

## Reproducibility

The source scripts are:

- `scripts/plot_cxt_fish_manuscript_figures.py`
- `scripts/build_final_cxt_fish_manuscript.py`

The final artifacts and their hashes are recorded in `reports/cxt_fish_submission_v5_visual_audit.md`.
