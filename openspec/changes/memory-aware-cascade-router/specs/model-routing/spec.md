## ADDED Requirements

### Requirement: Capability-safe routing
The router MUST select only models with the modality required by the task.

#### Scenario: Scene inference fails twice
- **WHEN** the visual production model and its constrained retry fail
- **THEN** the task is marked for human review and no text-only model is used

#### Scenario: Semantic generation fails validation
- **WHEN** constrained retry also fails
- **THEN** deterministic facts and a safe template are returned

### Requirement: Memory-aware cascade
The router MUST consider lifecycle state and quality before model execution.

#### Scenario: Fresh high-quality memory exists
- **WHEN** active memory has quality at least 0.9
- **THEN** the route action is `reuse_memory`

#### Scenario: Memory is incomplete
- **WHEN** useful fields exist but required fields are missing
- **THEN** the route action is `incremental_inference`

### Requirement: Shadow isolation
Shadow results MUST NOT affect production output or the memory merge.

#### Scenario: A Shadow candidate is ready
- **WHEN** Shadow mode is enabled
- **THEN** it runs separately and writes a `.shadow.json` result

#### Scenario: A Shadow candidate is unavailable or fails
- **WHEN** its model or adapter is missing, or inference raises an error
- **THEN** production execution continues unchanged and the reason is logged

### Requirement: Explainable decisions
Every route MUST expose its action, selected model identifiers, reasons, and safeguards.

#### Scenario: A route is inspected
- **WHEN** the user runs `rcpp router explain`
- **THEN** the CLI emits the routing context and complete decision as JSON

### Requirement: Provider credential isolation
General-reasoning calls MUST use only credentials belonging to the selected Provider.

#### Scenario: Only a DeepSeek key exists for an OpenAI route
- **WHEN** the selected Provider is OpenAI
- **THEN** the route reports missing OpenAI credentials and does not reuse the DeepSeek key
