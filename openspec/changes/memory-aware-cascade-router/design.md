# Design

The router emits a decision and reasons; the inference layer executes it. Fresh,
high-quality memory can bypass inference. Partial memory triggers incremental work.
Conflicts and expiry force refresh. Scene failures can reach human review but cannot
fall through to a text-only model.

Shadow candidates run sequentially when ready. Their files use a `.shadow.json` suffix
and never enter the merge step. Missing candidates are skipped without failing production.
