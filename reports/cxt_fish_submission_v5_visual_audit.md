# CXT-Fish submission v5 visual audit

## Audited artifacts

- DOCX: `paper/CXT-Fish_IMTS_Submission_Ready_v5.docx`
- DOCX SHA-256: `A7B8C76BCFC7238E50AE0E0F62E8E85F9E4D4AF06A512D2FB41D500C63781930`
- PDF: `paper/CXT-Fish_IMTS_Submission_Ready_v5.pdf`
- PDF SHA-256: `C0B1056BF76B014D3769F2258D8C9B9198F51CDDDE8AF5DAF3A843AD3E8D79FB`
- Rendered page count: 18
- Rendering route: Microsoft Word PDF export followed by Poppler page rendering at 140 dpi. The bundled LibreOffice renderer was unavailable on this host.

## Page-by-page result

All 18 final PDF pages were inspected. Pages 1–12 were pixel-identical to the previously inspected render after the final Table 8 pagination change; pages 13–18 were re-inspected after that change.

The audit found:

- no clipped figures, tables, captions, equations, or page numbers;
- no orphan section heading for Section 3.10;
- Tables 3 and 7 remain intact on one page;
- Table 8 now starts with Section 4.9 and is no longer split with an isolated repeated header row;
- Table 9 continues cleanly across pages 16–17 with repeated header structure;
- no figure-caption separation that changes interpretation;
- no visible overlap, raster corruption, or missing panel label;
- the correspondence email is `xiao.qin@sdust.edu.cn`;
- reference order is alphabetical by first author.

## Figure audit

Figures 1–8 use a common restrained journal visual system: white background, thin black/grey structure, limited colour accents, consistent panel labels, and data-first encoding. Figures 2, 3, 6, 7, and 8 are generated from frozen CSV/JSON artifacts. Figures 1, 4, and 5 are reproducible vector-style schematics and do not use generated or simulated experimental observations. Figure 4 remains explicitly schematic and does not redistribute raw Fish4Knowledge images.

## Conclusion

The v5 DOCX and PDF pass the final visual-layout audit. No scientific result, experimental definition, model output, or claim boundary was changed during this formatting pass.
