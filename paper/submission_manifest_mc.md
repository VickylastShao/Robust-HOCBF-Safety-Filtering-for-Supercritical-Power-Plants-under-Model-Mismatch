# M&C Major-Revision Submission Manifest

Manuscript ID: `MAC-26-0207`
Revision due date: 15 September 2026

The current upload-ready files are listed in [`revision_submission/README.md`](revision_submission/README.md). That directory is the only current revision package; the portal-downloaded files at the repository root and under `paper/` remain immutable submitted baselines for comparison.

## Primary Upload Files

| ScholarOne purpose | File |
|---|---|
| Revised title page | `revision_submission/Title_Page_revised.docx` |
| Highlighted revised main manuscript | `revision_submission/manuscript_mc_revised_highlighted.docx` |
| Supplemental material | `revision_submission/manuscript_mc_supplementary_revised.docx` |
| Response to Editor and Reviewers | `revision_submission/response_to_reviewers_mc.docx` |

The clean revised manuscript is an internal comparison copy and is not the highlighted upload manuscript.

## Separate Figure Files

All artwork files are 600 dpi RGB LZW TIFFs in `revision_submission/figures_for_upload/`:

1. `Figure_1_architecture.tif`
2. `Figure_2_process_response.tif`
3. `Figure_3_model_mismatch.tif`
4. `Figure_4_kappa_calibration.tif`
5. `Figure_5_gp_data_sensitivity.tif`
6. `Figure_6_historian_envelope.tif`
7. `Figure_7_retrofit_pair.tif`
8. `Figure_8_controller_logs.tif`

## Evidence Boundary

The revision uses the certificate-aligned drift-only benchmark and the current major-revision result inventories documented in `../ARTIFACT_MANIFEST.md`. Confirmed original plant-controller exports support the reported controller status, margins, saturation flags, and timing summaries. Deployment supervisor and DCS/Kubernetes details are configuration specifications, not individual export fields.

Do not upload internal audits, raw enterprise data, old `results/phase5/` development outputs, LaTeX build intermediates, or the unmarked internal DOCX as substitutes for the files above.
