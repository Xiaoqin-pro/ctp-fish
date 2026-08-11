# CXT-Fish v6 revision log

This revision addresses the final construct-validity and reproducibility review without training, re-inference, changing a checkpoint, or changing a reported number.

| Review point | v6 action |
|---|---|
| Tail-F1 tier definition was described as one frozen development mapping | Methods now state that tiers are constructed fold-locally from each outer-training partition; Supplementary Table S2b lists all three mappings. |
| Public split-manifest wording was too strong | Data/code availability now promises deterministic split reconstruction and hashes, not redistribution of ignored split files. |
| Absolute donor paths were not portable | Exported the exact primary and five sensitivity pairings as `reports/portable_frozen_donor_manifest.csv` with normalized relative IDs, stable hashes, and a resolver note. |
| Reviewer-facing package labels were visible in formal manuscript tables | Replaced them with `post-hoc validity controls` and `validity-control protocol`; historical report filenames and commit identifiers were not rewritten. |
| Fig. 2(c) did not identify its feature source | Caption now identifies the panel as a deep-feature audit and points to the pHash result in the text. |
| Favourable-cell counts could be over-read as independent trials | Main sensitivity prose and Table 6 explicitly mark these counts as descriptive. |
| Table 6 difference provenance | Caption now states that differences use unrounded cell-level values. |

Final verification: 88 tests passed; v6 DOCX/PDF and Supplementary DOCX/PDF were rendered and inspected; official Fish4Knowledge TEST remained unaccessed.
