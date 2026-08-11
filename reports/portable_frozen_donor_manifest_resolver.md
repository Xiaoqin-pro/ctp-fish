# Portable frozen donor manifest

`portable_frozen_donor_manifest.csv` is a path-independent export of the
frozen primary donor realization (seed 3407) and the five deterministic donor
sensitivity realizations (seeds 4101--4105). It contains the exact pairing
rows used in the saved evaluations, but no machine-specific absolute paths.

`recipient_relative_id` and `donor_relative_id` are normalized paths relative
to the local Fish4Knowledge root. `recipient_stable_id` and `donor_stable_id`
are SHA-256 hashes of those normalized relative identifiers. To resolve a row
on another authorized copy of the data, prepend that copy's Fish4Knowledge
root to the corresponding relative identifier and verify the stable hash.
The export does not alter or regenerate any frozen pairing.

The raw images and masks remain subject to the original Fish4Knowledge access
terms and are not redistributed by this repository.
