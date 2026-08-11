# Frozen donor protocol audit

This audit describes, but does not modify, the primary seed-3407
context-swap manifests. Each recipient has one same-class and one
cross-class donor from the same held-out outer fold. Donor candidates
are ordered deterministically and selected by the frozen hash rule.
The cross-class pool is not species-balanced before selection; its
frequency is therefore reported rather than hidden.

## Fold 1

- Rows: 17544; recipients: 8772
- Manifest SHA-256: `98e12cb69d4b36cc2c1f673f7704f326e1c8cbda07282fade11672157df76941`
- Cross-class donor species: 16
- Cross-class donor frequency range: 0.0019--0.2762
- Maximum cross-class donor-image reuse: 8
- Maximum cross-class donor-group reuse: 77

## Fold 2

- Rows: 18754; recipients: 9377
- Manifest SHA-256: `c8ae9421596cc202b60d040860dc42a4595338ef370ce56063313a9aedc934be`
- Cross-class donor species: 16
- Cross-class donor frequency range: 0.0016--0.2805
- Maximum cross-class donor-image reuse: 8
- Maximum cross-class donor-group reuse: 99

## Fold 3

- Rows: 17968; recipients: 8984
- Manifest SHA-256: `de042addd93048988461bb9777ba6597532f7306bd087cf96ab8599e5df74738`
- Cross-class donor species: 16
- Cross-class donor frequency range: 0.0019--0.2777
- Maximum cross-class donor-image reuse: 9
- Maximum cross-class donor-group reuse: 120

## Fixed semantics

The manifests are shared by F0 and F1 within each fold. Donor selection
uses the frozen seed 3407 and does not first balance donor species. The
composites therefore represent controlled conflicting donor-context
interventions, not guaranteed fish-free background replacement.
