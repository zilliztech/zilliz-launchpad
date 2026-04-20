## ADDED Requirements

### Requirement: Skill entry point

The skill SHALL expose a single entry file `skills/zilliz-launchpad/SKILL.md` that the agent loads when the skill is invoked. The file MUST NOT exceed 500 lines and MUST declare the skill name, a one-sentence description, and the ordered Phase 1–4 flow.

#### Scenario: Skill loads under the size budget

- **WHEN** the skill is installed via `npx skills add zilliztech/zilliz-launchpad`
- **THEN** `SKILL.md` is present in the installed skill directory
- **AND** `wc -l SKILL.md` returns a value `<= 500`

#### Scenario: Skill declares all four MVP phases

- **WHEN** `SKILL.md` is parsed for phase headings
- **THEN** sections for Collect, Configure, Plan, and Execute appear in that order

### Requirement: Phase flow contract

The skill SHALL execute Phases 1–4 in order. Each phase MUST receive its predecessor's output as input and MUST NOT start until the predecessor has produced its declared artifact. If any phase fails, the skill SHALL stop and surface the error with the failing phase identified.

#### Scenario: Phase 2 starts only after Phase 1 output exists

- **WHEN** Phase 1 (Collect) has produced a sample-shape analysis
- **THEN** Phase 2 (Configure) begins with that analysis available

#### Scenario: Failure in Phase 3 halts Phase 4

- **WHEN** Phase 3 (Plan) exits with a non-zero status
- **THEN** Phase 4 (Execute) does not run
- **AND** the agent reports which phase failed and the error

### Requirement: CLI command surface

The skill SHALL back every phase with a subcommand of `zilliz_ops.py` (a Typer-based CLI). The CLI MUST be invokable outside an agent and MUST exit with a non-zero status on failure. Phase-to-subcommand mapping MUST be 1:1 for Phases 1–4.

#### Scenario: Each phase has a CLI subcommand

- **WHEN** `python scripts/zilliz_ops.py --help` is run
- **THEN** subcommands `collect`, `configure`, `plan`, and `execute` are listed

#### Scenario: CLI is usable without an agent

- **WHEN** a developer runs `python scripts/zilliz_ops.py execute --plan runs/2026-04-20-001/plan.json` directly in a shell
- **THEN** the command runs to completion or exits non-zero with a descriptive error

### Requirement: Run output directory

Each invocation that produces a plan SHALL write to a new directory under `scripts/runs/<timestamp>/`. The directory MUST contain at minimum `plan.json` (machine-readable) and `plan.md` (human-readable). Timestamps MUST be UTC and lexicographically sortable (e.g., `2026-04-20T14-32-05Z`).

#### Scenario: Plan phase writes both formats

- **WHEN** `zilliz_ops.py plan` completes successfully
- **THEN** `scripts/runs/<timestamp>/plan.json` exists
- **AND** `scripts/runs/<timestamp>/plan.md` exists
- **AND** `plan.json` parses as valid JSON against the documented plan schema

### Requirement: Credential resolution

The skill and CLI SHALL resolve all secrets (Zilliz Cloud token, embedding provider API keys) from environment variables first. If a required variable is unset while running under the skill (not direct CLI use), the CLI SHALL exit with a structured error that names the missing variable and an example `export` command, so the agent can prompt the user precisely.

#### Scenario: Credential present in env

- **WHEN** `OPENAI_API_KEY` is set in the environment
- **AND** any phase needing it runs
- **THEN** the phase proceeds without prompting

#### Scenario: Credential missing under skill

- **WHEN** `OPENAI_API_KEY` is unset
- **AND** Phase 4 (Execute) needs to embed data
- **THEN** the CLI exits non-zero with an error payload containing the variable name and `export OPENAI_API_KEY=...` hint
- **AND** the skill relays this to the user as a dialogue prompt

### Requirement: Reference lazy-loading

Phase-specific reference material SHALL live under `skills/zilliz-launchpad/references/` and MUST only be loaded into agent context for the active phase. `SKILL.md` MUST document which reference files belong to which phase.

#### Scenario: Only current phase's references are loaded

- **WHEN** the agent is executing Phase 3 (Plan)
- **THEN** `references/knowledge/index_tuning.md` and peers are loaded
- **AND** `references/deploy-serverless.md` (Phase 6 material) is not loaded
