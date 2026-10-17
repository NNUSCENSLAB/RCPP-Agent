# Durable planning and measured budget routing

Retain LangGraph and replace the supervisor routing hub with explicit stage edges.
Introduce per-RPS refresh receipts, a resumable human Scene review boundary, and
model admission based on approved measured quality, cost and latency profiles.

Scope: runtime, memory refresh, graph-write atomicity, router, CLI and regression tests.
No new model training, vector retrieval or planning algorithm changes are included.
