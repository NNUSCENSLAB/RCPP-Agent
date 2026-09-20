# Proposal: CLI-only runtime

The interactive Python script hid configuration, checkpoints, and failure status.
Expose one installable `rcpp` command with structured subcommands and stable exit codes.

The workflow remains importable from `rcpp_core.runtime` for tests. The former
`src/rcpp_agent.py` entrypoint is deleted.
