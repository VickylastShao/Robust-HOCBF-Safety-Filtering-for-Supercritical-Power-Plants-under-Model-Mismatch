# M&C Submission Closeout Plan

Date: 2026-06-30

Target venue: Measurement and Control (SAGE)

Submission line:

- Main manuscript: `paper/manuscript_mc.tex`
- Main PDF: `paper/manuscript_mc.pdf`
- Supplement: `paper/manuscript_mc_supplementary.tex`
- Cover letter: `paper/cover_letter_mc.tex`

## Objective

Bring the Measurement and Control submission package to a defensible
pre-submission state by eliminating target-venue ambiguity, unsupported claims,
formal-scope overstatements, and LaTeX/package defects.

## Execution Tasks

1. Create a venue-compliance and submission-package audit.
   - Record M&C requirements from the SAGE author instructions.
   - Freeze M&C files as the only submission package.
   - Mark JPC/TAC files as archive/reference only.

2. Create a claim-evidence matrix.
   - Trace every abstract/conclusion/table-caption numerical claim to result
     files or generated analysis outputs.
   - Mark claims as supported, needs verification, or unsafe.
   - Do not introduce new claims without a source.

3. Patch hard-risk manuscript wording.
   - Remove any statement that the full margin makes the QP always feasible.
   - Clarify the difference between formal certificate scope and empirical
     margin calibration.
   - Align epsilon terminology across abstract, methods, and conclusions.

4. Run result inventory checks.
   - Phase 5 strict audit must pass.
   - Phase 4 audit may remain non-strict if Phase 4 is not part of the M&C
     evidence base; document the gap.

5. Compile LaTeX outputs.
   - Compile `manuscript_mc.tex`.
   - Compile `manuscript_mc_supplementary.tex`.
   - Inspect logs for fatal errors, undefined references, missing citations,
     and major overfull warnings.

6. Create final submission manifest and residual-risk report.
   - List exactly what to upload.
   - List what not to upload.
   - List any residual decisions requiring author confirmation.

## Stop Conditions

Do not declare submission-ready if:

- The main manuscript does not compile.
- There are undefined references or missing citations.
- A headline numerical claim cannot be traced to evidence.
- The formal guarantee claims exceed the assumptions and feasibility conditions.
- The M&C package still mixes JPC/TAC response material into the upload set.
