# Runtime reliability

## Requirement: Explicit planning order

Task initialization SHALL run once. Planning stages SHALL follow direct edges.
Review feedback SHALL repeat weighting, phased decisions, memory update and review only.

## Requirement: Per-RPS recovery

Each returned Scene/Semantic extraction SHALL be committed to a run-scoped receipt
before graph write. Recovery SHALL reuse committed matching receipts. Input, image,
model, Adapter and relevant code changes SHALL invalidate receipt identity.
Memory version, facts and image relations SHALL commit in one graph transaction.
Replay of a superseded memory ID SHALL NOT demote the newer current memory.

An interrupted uncommitted provider request MAY be repeated. The system SHALL NOT
claim exactly-once billing without provider idempotency support.

## Requirement: Human review

Invalid Scene after constrained extraction retry SHALL interrupt the graph before
suitability. CLI status SHALL be awaiting_review. A submission SHALL identify the
reviewer, match the pending token and cover each pending RPS exactly once. It SHALL
either exclude the RPS for this run or provide a valid Scene correction with image
evidence. Recovery SHALL recheck Scene before querying Semantic.

## Requirement: Measured model admission

Budgeted routing SHALL use approved measured profiles matching the deployed model
and Adapter. Shadow or cross-modality deployments SHALL NOT be promoted by routing.
Quality admission SHALL precede cost minimization. A missing or infeasible profile
SHALL produce human review for Scene or deterministic rules for Semantic.

Latency P95 and mean cost SHALL be described as statistical selection constraints.
The system SHALL NOT claim a hard runtime or billing cap based on these statistics.

## Requirement: Checkpoint compatibility

The runtime SHALL reject automatic resume of checkpoints from the previous graph
topology. Historical checkpoint files SHALL remain available for the previous runtime.
