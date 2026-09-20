## ADDED Requirements

### Requirement: Separate memory types
Scene and Semantic memory MUST remain separate and MUST be associated through the same
`rps_id`.

#### Scenario: A complete RPS memory is persisted
- **WHEN** both extraction results pass validation
- **THEN** Neo4j stores independent `SceneMemory` and `SemanticMemory` nodes linked to the RPS

### Requirement: Versioned lifecycle
Replacing a memory MUST preserve the prior version and its provenance.

#### Scenario: A newer memory version is written
- **WHEN** an active version already exists for the same RPS and memory type
- **THEN** the old version is marked `superseded` and the new version links to it with `SUPERSEDES`

### Requirement: Safe memory reuse
Only active, unexpired memory meeting the quality threshold MUST be directly reusable.

#### Scenario: Memory is fresh and high quality
- **WHEN** status is active, expiry has not passed, and quality is at least 0.9
- **THEN** the runtime may bypass extraction inference

#### Scenario: Memory has a conflict or has expired
- **WHEN** status is conflicted or the expiry time has passed
- **THEN** the runtime refreshes the memory and retains the previous evidence

### Requirement: Deterministic Semantic facts
Semantic numeric facts and rule labels MUST remain authoritative over generated prose.

#### Scenario: Generated prose conflicts with a rule fact
- **WHEN** consistency validation detects the conflict
- **THEN** the generated result is rejected and the deterministic fallback is retained
