# Scene–Semantic graph long-term memory specification

### Requirement: Single durable memory source

The runtime MUST read and write long-term Scene and Semantic memory through Neo4j. It MUST NOT
read or write a `GlobalSceneSemanticMemory` aggregate.

### Requirement: Independent immutable versions

Each Scene and Semantic result MUST have a content-specific `memory_id`. Repeating a write with
the same ID MUST be idempotent. Appending a new version MUST mark the prior current version as
superseded, and each RPS MUST have at most one current version per memory kind.

### Requirement: Scene→Semantic progressive retrieval

Semantic memory MUST be queried only for RPS IDs whose current Scene memory is valid and has
`clearance_visual_assessment=true`, `has_existing_RCP=false`, and
`is_functional_zone=false`. Ground obstacles MUST NOT stop retrieval when clearance remains true.

### Requirement: Invalid memory handling

Missing fields and type errors MUST produce `partial`. Cross-field contradictions and multiple
current versions MUST produce `conflicted`. Neither state may pass automatically.

### Requirement: Evidence trace

Every persisted planning decision MUST link to its RPS and PlanningRun, applied rules, and
supporting Scene or Semantic facts. The CLI MUST expose the resulting evidence path.

### Requirement: Audited migration

The migration MUST support dry-run and apply modes. Scene records MUST be joined by exact image
basename; unmatched records MUST remain unbound and appear in the audit report. Apply mode MUST
delete the legacy aggregate after normalized writes complete.
