# Plan

Milestones mapped to the ship gates of docs/BUILD_BRIEF.md section 6, with risks, fallbacks that
depend on the environment, and usage estimates. Environment facts come from `make doctor`
(STATUS.md). Usage figures below are planning estimates, not results.

## Milestones

| # | Milestone | Gates | Tag |
|---|-----------|-------|-----|
| M0 | Bootstrap: brief, doctor, scaffold, make targets, ship-check, review tooling | (all wired) | |
| M1 | evalstats core: long-format schema, generic CSV/JSONL reader, paired bootstrap, cluster-robust SEs, analytic power and MDE, CLI `compare` and `power`; tests to 85% | G1, G2, G3 | |
| M2 | Offline demo on recorded, labeled fixtures (synthetic eval results plus a small recorded leaderboard sample) with golden output | G4 | v0.1 |
| M3 | Integrity tooling: prereg hash chain, claimcheck, freshness (ported from raise-v1 with attribution), citecheck validator, citation-checker run | G10, G11 (partial G13) | |
| M4 | E0: freeze the simulation grid in prereg, coverage simulations, statsmodels cross-checks, hypothesis suite, unsupported-cell warnings | G5 | |
| M5 | GLMM and variance partitioning; rank-confidence sets, Bradley-Terry, IRT, partial-identification bounds; orchestrator with planted-component recovery on 200+ seeded datasets | G6 | |
| M6 | Subjects and suite: local model runtime, Docker-free agentic suite, sandbox, client wrapper with caps and ledger; pilot on disjoint tasks to put each configuration at 0.2 to 0.8 success | (G7 prerequisite) | |
| M7 | E1 preregistration, then confirmatory runs over at least 3 days; Claude arm where the billing guard is clean | G7 | v0.5 |
| M8 | E2 reanalysis of at least 2 leaderboards with pinned snapshots; E3 policy-sensitivity bounds; confidence reports | G8 | |
| M9 | Wheel, 5 runnable examples, generated API reference; README per section 11; REPORT.md | G9, G13 | |
| M10 | Review, hygiene, final sequence (section 10.6) | G12, G14 | v1.0 |

Order on the critical path: M1, M2, M3 and M4 are fully local and unblock G1 to G5 and G10 to
G11 early. M6 starts as soon as M1 is stable because E1 needs calendar days (3 or more distinct
days) and the Claude arm is rate-capped per day.

## Risks and fallbacks

| Risk | Fallback |
|------|----------|
| Billing guard trips here (hosted session sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`) so no Claude children can launch | Claude subject arm and headless review run only on a machine where `make doctor` shows the guard clean (Raj's laptop). Reviews use the `reviewer` subagent path. G7's 150 Claude runs are listed under Needs Raj until then. |
| No local model runtime (no Ollama, no GPU, 4 CPUs) | Install a CPU runtime (Ollama or llama.cpp) if the network policy allows; pick small pinned GGUF models (about 0.5B to 4B) so many runs fit in CPU time; record file hashes. If nothing installs, list it under Needs Raj. |
| No Docker daemon | Docker-free suite only; Terminal-Bench through Harbor is skipped and recorded in DECISIONS.md. Sandbox through Linux namespaces (`unshare`), with mount confinement to a temp workspace. |
| No R, so no lme4 | Try installing R and lme4; otherwise use an established Python GLMM backend, name it, and validate its intervals in E0. Variance-component intervals by parametric bootstrap, not Wald. |
| CPU time for E1 | Fractional factorial with blocking; power analysis under the caps sets the sample size; runs in the background with PID files and resumable caching. |
| Public per-task leaderboard data and licenses | Verify per-task availability and licenses before choosing boards; store only what licenses allow, hashes for the rest. |
| Usage limits on the shared plan | Stop launching at a usage-limit message, record the reset time, resume on the next heartbeat. |

## Usage estimates (planning only)

- Claude subject runs (E1 Claude arm): at least 150 multi-turn runs at no more than 30 per day
  (times `CAP_SHARE`), so at least 5 days of the arm; more if the preregistered power analysis asks.
- Claude single calls: pipeline labeling and extraction, well under the daily cap.
- Reviews: one per milestone, at most 6 per day.
- Local model runs: the bulk of E1 cells (several hundred to a few thousand short agent runs),
  bounded by CPU time rather than any cap.
- Cash: 0.

## Claude arm on Raj's machine (G7)

The Claude tiers of E1 run through headless Claude Code on a machine where the billing guard is
clean (D-004, D-019). This hosted session cannot run them by design. Steps for Raj:

1. Clone the repository at the commit named in the E1 preregistration, then `make setup` and
   `make doctor`. Its `billing guard` row must read `clean`: no `ANTHROPIC_API_KEY`,
   `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_*`, or `apiKeyHelper` in any
   user, project, or managed settings file. The `claude` CLI must be logged in to the Max plan
   (check with `/status` inside `claude`). Turn off paid extra usage in the plan settings for the
   duration of the arm; the client stops at once and refuses further runs if the CLI reports
   extra usage in use or a refused plan limit.
2. Smoke check (one exploratory run, counts as one `agent_run`):
   `uv run python -m noisefloor.claude_arm smoke --tier claude-haiku`. It must end with
   `SMOKE: PASS`, which means the child reported the plan login (`apiKeySource none`), only the
   file tools Read, Glob, Grep, Write, and Edit, no MCP servers, and no memory files. Anything
   else prints `SMOKE: CHECK <what>`: stop and report it.
3. Run the arm with the command the E1 preregistration names. It is resumable: rerun the same
   command each day. Daily caps come from budget.toml times `CAP_SHARE`; at `CAP_SHARE = 0.2`
   that is 6 runs a day (about 25 days for 150 runs), at `CAP_SHARE = 1` it is 30 a day (about 5
   days). The turn cap per run is `claude_agent_max_turns` times `CAP_SHARE` (8 at 0.2).
4. Commit `artifacts/e1/claude.jsonl` and `artifacts/usage/ledger.jsonl` (the ledger logs tokens
   and `plan_equivalent_usd`, an API-price estimate, never cash), and push to the session branch
   or send the files back.
