# Design

`GraphMemoryRepository` is the single durable-memory boundary. `RPS` nodes link to
independently versioned `SceneMemory` and `SemanticMemory` nodes. Each memory contains
structured Fact nodes and provenance. New versions supersede the previous current version.

Retrieval first loads current Scene memories and validates schema, lifecycle, and the three
physical conditions. Only IDs that pass are included in the Semantic query. A failed physical
condition creates an evidence-backed Decision. Missing, stale, partial, and conflicted memories
produce explicit refresh or review events.

Planning decisions link to their RPS, PlanningRun, PlanningRule, and supporting Fact nodes.
This supplies a traversable evidence path without constructing a second aggregate memory store.

Migration joins original Scene inference to final RPS records by exact image basename. It never
uses coordinates as an implicit join key. Unmatched records remain in the audit report.
