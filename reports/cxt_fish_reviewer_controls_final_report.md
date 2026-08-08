# CXT-Fish reviewer-control package: final report

This closed package was completed after the frozen outer results. It is
post-hoc reviewer-motivated evidence, not a new method-selection stage.

## Completed controls

- A: class-inclusion, donor-protocol, and frozen-artifact audits.
- B: five alternative donor realizations (4101--4105), 90/90 cell evaluations.
- C: F0-2RGB supervision-matched control, 9/9 training and 9/9 evaluation cells.
- Official Fish4Knowledge TEST: not accessed.

## B result

The five alternative donor realizations all produced positive equal-weight
cell-mean CXT-Fish minus F0 cross-composite differences. The realization
range and cell-level counts are in `reports/cxt_fish_donor_sensitivity_report.md`.
The original seed-3407 +7.24pp result remains the frozen primary result.

## C result

F0-2RGB has the same ResNet18, sampler, optimizer budget, 2B forward, and
two CE terms, but uses two independently augmented ordinary-RGB views with
no mask or foreground view. The exploratory paired bootstrap and summary
are in `experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000.json` and
`reports/cxt_fish_rgb2_control_report.md`.

## Interpretation boundary

These controls address duplicated supervised exposure and donor-assignment
concerns. They do not establish cross-dataset, cross-camera, or cross-site
generalization. They do not convert the post-hoc controls into confirmatory
endpoints or alter the frozen primary result.

Protocol commit: `11ecb6436e78af9a22c5141760ff7dceefc60bab`
Final commit: `61b36a4fd833792b177c84c54b4f0db5183b8cb5`

Integrity audit: passed.
