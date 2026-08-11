# CXT-Fish submission v2 visual revision log

Date: 2026-08-09

This revision is a visual and packaging rewrite only. No model was trained, no
prediction was regenerated, no split was changed, and no scientific value or
claim was changed.

## Figure system

- Rebuilt all eight manuscript figures from the frozen CSV/JSON artifacts.
- Replaced the previous presentation-like visual language with a restrained
  journal system: white background, compact multi-panel layouts, direct labels,
  limited colour, light grid lines, and vector primitives for schematics.
- Exported every figure as editable SVG and PDF, plus 600-dpi PNG.
- Kept all schematic fish symbols explicitly non-photorealistic; no generated
  fish imagery or redistributed raw Fish4Knowledge image was introduced.
- Figure sources remain reproducible in
  `scripts/plot_cxt_fish_manuscript_figures.py`.

## Article package

- New manuscript: `paper/CXT-Fish_IMTS_Submission_Ready_v2.docx`.
- New supplementary package: `paper/CXT-Fish_IMTS_Supplementary_Material_v2.docx`.
- Corresponding-author email: `xiao.qin@sdust.edu.cn`.
- The supplementary package keeps all evidence notes and captions while using
  a clean two-page layout.

## QA

- Full repository tests: 88 passed.
- Manuscript rendered with the installed Word engine: 17 pages inspected at
  120 dpi; no clipped figures, table overflow, or broken captions observed.
- Supplement rendered with the installed Word engine: 2 pages inspected at
  120 dpi; S4 is kept intact on one page and S8 retains its caption.

## External visual references consulted

The redesign follows the general principles documented by the public Nature
figure skill, SciencePlots, cnsplots, and AcademicForge scientific-visualization
guidance. IMTS examples were checked for their compact systems schematics and
clear multi-panel data figures. These references informed visual conventions
only; no third-party artwork was copied into the manuscript.
