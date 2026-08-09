# Final manuscript consistency audit

Audit target: `paper/CXT-Fish_IMTS_Final_ConstructValidity_Manuscript_revised.docx`

## Frozen numerical checks

| Quantity | Manuscript value | Frozen artifact | Status |
|---|---:|---|---|
| F4K-16T images/species/groups | 27,133 / 16 / 8,680 | dataset and outer manifests | PASS |
| Gate-0 image-level macro-F1 | 0.9917 | `experiments/ctp_fish_gate0_summary.csv` | PASS |
| Gate-0 group-disjoint macro-F1 | 0.9669 | `experiments/ctp_fish_gate0_summary.csv` | PASS |
| Gate-0 difference | −2.48 pp | derived from frozen Gate-0 summary | PASS |
| Main F0 clean macro-F1 | 0.9577 | `experiments/cxt_fish_final_method_summary.csv` | PASS |
| Main CXT-Fish clean macro-F1 | 0.9556 | `experiments/cxt_fish_final_method_summary.csv` | PASS |
| Main clean difference | −0.21 pp | equal-weight nine-cell difference | PASS |
| Main F0 cross composite macro-F1 | 0.5189 | frozen main summary | PASS |
| Main CXT-Fish cross composite macro-F1 | 0.5913 | frozen main summary | PASS |
| Main cross-composite difference | +7.24 pp | `cxt_fish_resnet_outer_bootstrap_5000.json` | PASS |
| Main conditional interval | +6.13 to +8.20 pp | `cxt_fish_resnet_outer_bootstrap_5000.json` | PASS |
| F0-2RGB corrected difference | +6.83 pp | `cxt_fish_rgb2_vs_f1_bootstrap_5000_v2.json` | PASS |
| F0-2RGB exploratory interval | +5.71 to +7.87 pp | frozen v2 JSON | PASS |
| Fish-suppressed difference | +10.48 pp | `cxt_fish_fish_suppressed_donor_bootstrap_5000.json` | PASS |
| Fish-suppressed exploratory interval | +9.38 to +11.61 pp | frozen JSON | PASS |
| MobileNet clean difference | −2.94 pp | frozen per-fold results | PASS |
| MobileNet cross-composite difference | +4.82 pp average; 2/3 folds positive | frozen per-fold results | PASS |
| Group-weighted cross-composite difference | +4.47 pp | new frozen-prediction analysis | PASS |
| Species–group-balanced cross-composite difference | +6.73 pp | new frozen-prediction analysis | PASS |
| Fold × species group strata | 48; range 5–1414 groups | new manifest audit | PASS |

## Evidence-role checks

- Main result is explicitly described as a same-corpus frozen group-disjoint re-evaluation.
- Development and outer folds are stated to come from the same Fish4Knowledge-derived corpus.
- Main interval is explicitly conditional and paired; post-hoc intervals are labelled exploratory.
- The final synthesis calls this a focal robustness contrast, not a manuscript-level primary robustness estimand.
- Donor-realization, F0-2RGB, donor-subject-suppressed, group-weighted, MobileNet, and Route C analyses are not presented as independent validation.
- Macro-F1 weighting and group-weighted sensitivity are distinguished.
- Training masks are identified as privileged annotations with preparation cost.
- Deployment claim is restricted to the recognition stage after fish crops are available.
- Camera/session/deployment/date independence is explicitly marked unverifiable from frozen recognition metadata.
- Official Fish4Knowledge TEST access is false throughout the evidence package.

## Terminology scan

The revised DOCX was scanned for misleading `post-development` or `confirmation` labels. Remaining uses of `independent validation` occur only in explicit negative statements (e.g., “not independent validation”), which is intentional.

## Figure and source checks

- Eight embedded figures were replaced with reproducible vector/synthetic schematics or CSV/JSON-derived matplotlib plots.
- No generated fish imagery is used.
- Figure source is `scripts/plot_cxt_fish_manuscript_figures.py`.
- Quantitative figure panels use frozen artifacts only.
- Fig. 1, Fig. 4, and Fig. 5 use white-background, thin-rule, low-saturation vector schematics with standard subpanel/stage labels; all figures have PNG/PDF/SVG exports.
- Supplementary Note S6 and Tables S7–S8 exist and match the manuscript's Route C, residual-bin, and donor-species references.

## Final reviewer judgement

The manuscript is internally consistent with the frozen evidence boundary. It supports a narrow construct-validity and targeted robustness claim, not external-dataset generalization, backbone-agnostic behavior, universal clean-accuracy improvement, causal background isolation, or superiority over CLIB.
