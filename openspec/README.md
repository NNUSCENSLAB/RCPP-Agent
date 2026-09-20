# RCPP-Agent OpenSpec

OpenSpec records behavior changes before implementation. Each change contains its
motivation, design constraints, testable requirements, and implementation checklist.
The specifications use RFC 2119 terms: **MUST**, **SHOULD**, and **MAY**.

Active changes:

- `cli-only-runtime`
- `memory-lifecycle-and-retrieval`
- `memory-aware-cascade-router`

Implementation status is tracked in each `tasks.md`. A checked task means code and
tests exist; it does not by itself mean a candidate model passed the promotion gate.
