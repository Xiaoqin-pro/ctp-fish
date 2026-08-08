# Manuscript terminology patch

Replace unqualified uses of `confirmation`, `confirmed result`, and
`confirmatory ResNet analysis` with `frozen post-development group-disjoint
evaluation`, `main frozen evaluation`, or `main frozen ResNet18 analysis`.

Retain `post-hoc`, `exploratory`, `pre-specified seeds`, and `frozen protocol`
where they are accurate. Keep these distinctions explicit:

- frozen is not preregistered;
- a pre-specified seed is not a preregistered endpoint;
- group-disjoint is not external validation;
- a post-development outer evaluation is not an independent confirmation.

The primary confidence intervals are conditional on the frozen trained models,
completed method development, and the frozen primary donor
manifest/intervention realization. They do not quantify optimization-seed
population uncertainty, method-development uncertainty, donor-family
uncertainty, or external-dataset uncertainty. The F0-2RGB interval is
exploratory/post-hoc; the fish-suppressed donor interval is a post-hoc
construct-validity sensitivity.
