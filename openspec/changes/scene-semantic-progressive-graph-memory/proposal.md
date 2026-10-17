# Proposal: Scene–Semantic progressive graph memory

## Motivation

The aggregate JSON memory duplicates normalized Neo4j state and prevents per-RPS retrieval,
version checks, and evidence tracing. Planning needs one authoritative long-term memory source.

## Change

- Make Neo4j the only durable Scene and Semantic memory source.
- Store immutable memory versions, facts, source evidence, rules, runs, and decisions as graph nodes.
- Apply Scene→Semantic progressive retrieval for each RPS.
- Migrate legacy JSON with exact image-name matching and an auditable dry-run.
- Expose memory inspection, history, trace, and route explanation through the CLI.

## Exclusions

This change does not add vector search, hypergraphs, GraphRAG, or model training.
