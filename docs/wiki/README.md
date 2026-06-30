# RoCBF-Net Paper Wiki

LLM-maintained knowledge base for the RoCBF-Net project. All content is generated and maintained by the LLM agent; the user sources papers and directs analysis.

## Structure

```
wiki/
├── index.md          # Content catalog — all pages with links and one-line summaries
├── log.md            # Chronological append-only log of wiki operations
├── papers/           # One page per paper, named by first-author-year
├── concepts/         # Cross-paper concept pages (e.g., HOCBF, GP-state-space)
└── comparisons/      # Comparison and synthesis pages across papers
```

## Page Format

Every page uses this frontmatter:

```yaml
---
title: Short Title
authors: [Author1, Author2]
year: 2024
tags: [tag1, tag2]
sources: [paper-page-ids that contribute to this page]
updated: 2026-05-19
---
```

## Paper Page Template

```markdown
---
title: Paper Short Title
authors: [Full Author Names]
year: YYYY
venue: Journal/Conference
tags: [relevant tags]
sources: [self]
updated: YYYY-MM-DD
---

## One-Line Summary
<1 sentence capturing the core contribution>

## Problem Setting
- System model: <dynamics form>
- Uncertainty model: <what kind of uncertainty>
- Control objective: <what is being optimized>

## Key Contributions
1. ...
2. ...

## Core Formulation
### <Main Method Name>
- Key equations (numbered, with variable definitions)
- Assumptions and their implications

## Theoretical Results
- Theorem statements (informal but precise)
- Key proofs sketch if important for implementation

## Algorithm / Implementation
- Pseudocode or algorithmic steps
- Computational complexity notes

## Limitations
- What the method cannot do
- Assumptions that may not hold in practice

## Relevance to RoCBF-Net
- Which phase(s) this paper feeds into
- Specific formulas/ideas we will adopt
- What we need to modify or extend

## Cross-References
- [[paper-page-id]] — relationship description
- [[concept-page-id]] — relationship description
```

## Concept Page Template

```markdown
---
title: Concept Name
tags: [tag1, tag2]
sources: [paper-page-ids that define/discuss this concept]
updated: YYYY-MM-DD
---

## Definition
Precise mathematical definition

## Variants Across Papers
| Paper | Variant | Key Difference |
|-------|---------|---------------|
| ...   | ...     | ...           |

## Implementation Notes
How this maps to code in RoCBF-Net

## Open Questions
Unresolved aspects relevant to our project
```

## Comparison Page Template

```markdown
---
title: Comparison: X vs Y
tags: [comparison]
sources: [paper-page-ids]
updated: YYYY-MM-DD
---

## Dimension 1: ...
## Dimension 2: ...
## Synthesis: What RoCBF-Net takes from each
```

## Operations

### Ingest
1. Read paper text (from pdftotext extraction)
2. Create paper page in `papers/`
3. Update `index.md`
4. Update or create relevant concept pages
5. Update or create comparison pages if multiple papers cover same topic
6. Append entry to `log.md`

### Query
1. Read `index.md` to find relevant pages
2. Drill into specific pages
3. Synthesize answer with wiki page citations
4. Optionally file the answer as a new wiki page

### Lint
1. Check for contradictions between pages
2. Find orphan pages (no inbound links)
3. Find missing concept pages (mentioned but not created)
4. Check stale claims against current paper understanding
