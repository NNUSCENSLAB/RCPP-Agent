## ADDED Requirements

### Requirement: One public executable
The installed package MUST expose `rcpp`, and the workflow module MUST remain importable
without exposing an interactive executable entrypoint.

#### Scenario: User inspects the CLI
- **WHEN** the user executes `rcpp --help`
- **THEN** the command lists doctor, data, memory, plan, router, and eval groups

#### Scenario: Legacy module is inspected
- **WHEN** the source tree is inspected
- **THEN** `src/rcpp_agent.py` does not exist and `rcpp_core.runtime` contains no interactive input loop

### Requirement: Deterministic structured path
The structured planning path MUST run without a general-model API key and MUST default
to cached AHP weights.

#### Scenario: No cloud credentials are configured
- **WHEN** a structured plan is started with an area and time horizon
- **THEN** instruction parsing and AHP weighting require no cloud language-model call

#### Scenario: A run fails
- **WHEN** workflow execution raises an exception
- **THEN** the CLI returns a non-zero exit code and records the error in its manifest

### Requirement: Resumability
A run MUST use its `run_id` as the LangGraph checkpoint thread identifier.

#### Scenario: A non-terminal checkpoint exists
- **WHEN** `rcpp plan resume <run-id>` is executed
- **THEN** the workflow resumes from that checkpoint
