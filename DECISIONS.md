# Decisions

Assumptions, substitutions, and interpretations, each recorded before acting on it.

## D-001 2026-10-23 Packaging
One `pyproject.toml` builds the `evalstats` distribution, whose wheel contains only
`src/evalstats`. The study package `src/noisefloor` sits beside it and is importable in the dev
environment through hatchling `dev-mode-dirs = ["src"]` in the editable install, but it is not
shipped in the wheel. Reason: the brief asks for `src/evalstats/` and `src/noisefloor/` and for a
pip-installable evalstats wheel (G9); this gives both without a second project file.

## D-002 2026-10-23 Python versions
Dev environment pinned to 3.12 (`.python-version`); CI runs 3.11 and 3.12 (G3); stdlib-only tools
(`tools/*.py`) need Python 3.11+ for `tomllib` and run through `uv run --no-project`.

## D-003 2026-10-23 Pushing from this cloud session (superseded by D-007)
Before the goal message arrived, `ALLOW_PUSH` was unset and pushes to the session branch were
justified by the session's own branch instruction. D-007 records Raj's explicit launch setting,
which replaces this interpretation.

## D-004 2026-10-23 Billing guard in the hosted environment
`make doctor` finds `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2` set by the hosting
environment, so the Claude backend is unusable here by the section 8 rule. No Claude child is
launched from this environment. `make review` uses the `reviewer` subagent path. The Claude
subject arm of E1 must run where the guard is clean.

## D-005 2026-10-23 Ship-check interpretations
Recorded in docs/GATES.md. In short: G1 core modules are all of both packages minus omissions
that have recorded-fixture contract tests, and line coverage is covered lines over statements;
G4 isolates the network with a network namespace when the OS allows it and says `env-only`
otherwise; G5 uses the observed coverage in the Monte Carlo standard error; G11 needs at least
one preregistration (no vacuous pass); G13 requires every external URL in README.md and
docs/REPORT.md, other than this repository, to be verified in `artifacts/citecheck.json`; G14
compares caps with the toml block in section 8 of the brief (`disk_floor_gb` may only rise). In
fast mode G3 infers `make ci` from its components and G7, G8, G13 skip recomputation; fast output
never counts.

## D-006 2026-10-23 Sandbox (pending)
No Docker daemon and no bubblewrap here; `unshare -rn` (user and network namespaces) works.
Network isolation is available; write confinement to a temporary workspace will be built with
mount namespaces before any subject agent or third-party code runs. Until a sandbox meeting
section 8 is verified, only this project's own generated tasks run.

## D-007 2026-10-23 Raj's launch settings for this web session
The `/goal` message sets, for this Claude Code on the web session, `ALLOW_PUSH=1` and
`CAP_SHARE=0.2` (five projects share one Max plan); the environment variables themselves are not
present in the container, so they are treated as set:
- Push every commit to the session's working branch `claude/gifted-darwin-k7q99b` on `origin`.
  Never force-push, never push tags, never change repository settings or visibility.
- `cap_share = 0.2` is recorded in budget.toml, so it does not depend on the shell. The effective
  share is the smaller of `cap_share` and the `CAP_SHARE` environment variable. Every `claude_*`
  cap is multiplied by it and rounded down: agent runs 6 per day, 8 turns per run (or per subtask
  message in resumed sessions), single calls 60 per day, review runs 1 per day. Applied literally
  to `claude_agent_max_turns` too; the E1 Claude arm will be designed within 8 turns per message.
- Reviews by the `reviewer` subagent count against `claude_review_runs_per_day` like headless
  reviews, so at most one `make review` per day.

## D-008 2026-10-23 R and lme4 in this container
`apt-get install r-base-core r-base-dev r-cran-lme4` works here (Ubuntu 24.04 archive). It is not
persistent: a new container needs it again, and Raj's environment setup script can add it. The
library keeps a named Python fallback so it works without R; `make doctor` reports which is
present, and every GLMM artifact records the backend that produced it.
