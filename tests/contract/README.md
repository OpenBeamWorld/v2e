# Connector contract tests

Tests every connector must pass: manifest and capability declarations, negotiation and refusal,
spawn at rest with requested versus actual pose recorded, actor-scoped actuation, the call
ledger and its offline replay audit. They run on the `fake` connector in CI; live connectors run
them only behind an opt-in marker.
