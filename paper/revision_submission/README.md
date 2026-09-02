# Measurement and Control Major-Revision Package

Manuscript ID: `MAC-26-0207`
Revision due date: 15 September 2026

## Upload Files

| ScholarOne purpose | File | Notes |
|---|---|---|
| Revised title page | `Title_Page_revised.docx` | Updated repository URL and current word/figure/table counts. |
| Highlighted revised main manuscript | `manuscript_mc_revised_highlighted.docx` | Editable Word document. Revised text and equations are marked with Word yellow highlighting; the reference list is unmarked. |
| Supplementary material | `manuscript_mc_supplementary_revised.docx` | Updated factual supporting material and proofs. |
| Response to Editor and Reviewers | `response_to_reviewers_mc.docx` | Point-by-point response. The same text can be pasted into the response field. |
| Figure 1 | `figures_for_upload/Figure_1_architecture.tif` | 600 dpi LZW TIFF. |
| Figure 2 | `figures_for_upload/Figure_2_process_response.tif` | 600 dpi LZW TIFF. |
| Figure 3 | `figures_for_upload/Figure_3_model_mismatch.tif` | 600 dpi LZW TIFF. |
| Figure 4 | `figures_for_upload/Figure_4_kappa_calibration.tif` | 600 dpi LZW TIFF. |
| Figure 5 | `figures_for_upload/Figure_5_gp_data_sensitivity.tif` | 600 dpi LZW TIFF. |
| Figure 6 | `figures_for_upload/Figure_6_historian_envelope.tif` | 600 dpi LZW TIFF. |
| Figure 7 | `figures_for_upload/Figure_7_retrofit_pair.tif` | 600 dpi LZW TIFF. |
| Figure 8 | `figures_for_upload/Figure_8_controller_logs.tif` | 600 dpi LZW TIFF. |

`manuscript_mc_revised_clean.docx` is the unmarked internal reference copy and should not replace the highlighted manuscript at upload. Re-upload `Title_Page_revised.docx` because its repository URL and manuscript counts have changed, even though the title, authorship, and affiliations are unchanged.

## Verification Record

- The package was regenerated from the current LaTeX and point-by-point response sources on 2 September 2026.
- The matching public artifact is identified by immutable tag `mc-major-revision-2026-09-02-resubmission-v2`.
- The revised title page is generated from the submitted layout by `scripts/build_revision_title_page.py`; its manuscript counts are computed from the clean revised DOCX and current LaTeX source.
- The Word manuscripts are produced from the current LaTeX sources by `scripts/build_mc_docx.sh`, then postprocessed for the M&C layout and metadata-cleaned.
- The submission PDFs were regenerated from the final DOCX files through Microsoft Word. A TeX engine was unavailable in the current build environment, so this revision record does not claim a fresh LaTeX-to-PDF compilation.
- The main manuscript contains the complete `Statements and Declarations` section required by the journal; author and affiliation details remain confined to the separate title page.
- The response letter preserves all 23 reviewer comments plus the Associate Editor's request, and every response identifies a section, figure, table, or supplemental location for verification.
- All figure upload files are RGB TIFFs rendered at 600 dpi and written with LZW compression.
- MW01--MW03 are indexed only as low-to-mid-load historian context; they are not used for controller-internal QP claims. The confirmed MW04--MW06 controller exports support the reported timestamps, operating states, margins, saturation flags, and timing summaries.
- The measured controller-export envelope reaches 629.7 MW (95.4% of the 660 MW nameplate rating). The manuscript and response explicitly avoid extrapolating this evidence to an unobserved 660 MW endpoint or calling it complete full-range validation.
- Public controller excerpts replace only the unit and internal controller-version identifiers; all other fields are copied verbatim and linked to source/public SHA-256 records.
- DOCX metadata fields `creator` and `lastModifiedBy` are empty, and public submission files contain no plant-specific unit name or internal controller-version string.
- The 1 s implementation is reported as the deployed controller-task period; the 5 s controller-export interval and the absence of a sampled-data certificate are stated separately.
