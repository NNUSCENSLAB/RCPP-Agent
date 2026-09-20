# Design

`rcpp_cli.app` parses commands and delegates to workflow/library functions. Every plan
run receives a `run_id` and writes a manifest under `.rcpp/runs/<run_id>`. Structured
arguments are the default path. `--instruction` is optional and is the only path that
requires a general-reasoning provider.

Cached AHP is the default so a missing cloud API key cannot block a structured run.
