# Design

Each RPS links to independently versioned `SceneMemory` and `SemanticMemory` nodes.
The active node supersedes the previous node without deleting it. Reuse requires an
active, unexpired memory whose quality meets the configured threshold. Invalid expiry
metadata is treated conservatively as a conflict.
