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

## E1 schedule and operations (this hosted session)

Order: freeze preregistrations 0004 (E1), 0005 (E0 part 2), and 0006 (G6) in one commit; run
G6 and E0 part 2 (synthetic, CPU only, Laplace bootstrap with task-jackknife refits at the
80-task E1 shape: about 8 core-minutes per dataset in a timing run, 1,200 GLMM datasets (G6 200,
E0 groups `e1` and `e1-het` 500 each), so about 40 hours on 4 workers, plus the cheap local-arm
and non-GLMM cells); bank both; then the E1 local arm (both arms refuse to start before the G6
and E0 part 2 artifacts are banked). Running G6 and E0 part 2 first lets a failing kind of
interval be replaced by a dated amendment with its own passing cells before any E1 data exist.
No commit changes `src/` or `uv.lock` from the start of G6 to the E1 analysis: the studies
refuse rows from a second environment, the arms stop on new run code, and the analysis runs
only under the analysis code G6 and E0 part 2 ran (D-033 item 10). Raw rows are committed at
check-ins so a lost container loses no banked work. Nothing else runs on the CPU while E1 runs,
because its runs have wall-clock limits.

- The local arm runs as `python -m noisefloor.e1 local prereg/0004-e1-factorial.md`, detached
  (`setsid nohup`, PID file and log under `runs/e1/`), with a watcher. It is resumable: a
  restart banks completed runs from the work directories and starts unfinished ones again.
- The first batch starts within 15 minutes after 00:00 UTC (launched later, the runner waits).
  A batch of slot day k, block b starts only in block b of the arm's day k or later, so the arm
  takes at least four UTC days (8 slots of about 9 hours at the dry run's rates, `e1-design`
  `duration_estimate`); slots that overrun move to the same block of the next day.
- Banked runs are committed locally after each check-in. Pushes (after `make prepush`, which
  needs the CPU) happen only while the arm waits between blocks. The same holds for tests,
  simulations, and anything else CPU-heavy in this session: CPU contention would lengthen runs
  and could turn some into timeouts, an effect outside the design.
- Check-ins keep the session awake while the arm runs; each one checks the process, the log,
  and the banked count, and relaunches the runner if it died. The log names banked and
  infra_error counts and infra_error end reasons only, never outcomes, and
  `artifacts/e1/local.jsonl` is read only by the runner until the arm ends.
- One runner at a time: each arm holds an exclusive lock (`artifacts/e1/.local.lock`,
  git-ignored); a second runner exits with code 4. Each batch runner holds a lock on its work
  directory, so a relaunch after a killed orchestrator waits for a live runner instead of
  running beside it, and a runner's model server stops when the runner dies.
- Stops (prereg 0004): a batch that fails as a whole (the model file fails its hash check, the
  server does not start or fails its health check after a run, three infra_error runs in a row,
  or the runner exits before every run is recorded) banks only the runs the failure did not
  touch; the others run again later under their planned run ids. The arm then stops, as it
  does when more than 5 percent (and at least 5) of the runs banked since its start or last
  resume are infra_error, and when a batch would start after day 7. Each stop is a line in
  `artifacts/e1/stops-local.jsonl` (reported as a deviation), and the runner exits with code 3
  until `python -m noisefloor.e1 resume local --note "<repair>"` records the repair. A batch
  that banks nothing because its calls were refused, or new run code (a commit under `src/`, a
  new lock, or another llama-server binary), also stops the arm; uncommitted changes under
  `src/` block every batch until they are committed. A defect found during the arm is recorded
  with `python -m noisefloor.e1 stop local --reason "..."` (it takes effect when the running
  batch ends) and resumes only with `--amendment` naming a frozen amendment of prereg 0004 that
  says how the affected runs are handled. The day-7 stop does not resume.

## Claude arm on Raj's machine (G7)

The Claude tiers of E1 run through headless Claude Code on a machine where the billing guard is
clean (D-004, D-019). This hosted session cannot run them by design. Steps for Raj:

1. Clone the repository (one clone only) at or after the commit that froze prereg/0004, with
   the G6 and E0 part 2 artifacts banked, then `make setup` and `make doctor`. Its `billing guard` row must read `clean`: no `ANTHROPIC_API_KEY`,
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
3. Run the slice with `uv run python -m noisefloor.e1 claude prereg/0004-e1-factorial.md`
   from that clone (the runner refuses a plan, pins, or tasks that differ from the frozen ones,
   a missing SMOKE: PASS, and uncommitted changes under `src/`; do not pull commits that change
   `src/` while the slice runs, or the slice stops until a resume accepts the new code as a
   deviation). It is resumable: rerun the same command each day. Daily caps are the
   budget.toml caps times the smaller of budget.toml's `cap_share` (0.2) and `CAP_SHARE`, so
   at most 6 runs a day (about 30 days for 180 runs); a faster slice needs `cap_share` raised
   in budget.toml by Raj (Claude never raises a cap). The turn cap per run is
   `claude_agent_max_turns` times that share (8 at 0.2); a run whose cap would be below its
   planned 8 steps is refused, so do not set `CAP_SHARE` below 0.2.
4. The slice ends when every planned run of the tiers still served is banked or 45 days after
   the freeze, whichever comes first; the runner then records an end point (run it once on or
   after day 45 if the slice has not finished), and its outcomes are not analyzed before then.
   An early end is `uv run python -m noisefloor.e1 end claude --reason "..."`, a deviation. Do
   not update Claude Code while it runs (its children run with `DISABLE_AUTOUPDATER=1`); a new
   CLI version stops the slice until a smoke run on it prints SMOKE: PASS and
   `uv run python -m noisefloor.e1 resume claude --note "<version change>"` accepts it as a
   deviation. If a tier is served by a model ID other than its first run's, that tier starts no
   further run and the others go on. A Claude Code child that did not start clean, or three
   infra_error runs in a row, stop the slice for repair. Any stop prints its reason and the
   resume command (exit code 3). Stops, resumes, and ends of the slice are recorded only on
   this machine (the commands refuse where the billing guard is not clean).
5. Commit `artifacts/e1/claude.jsonl`, `artifacts/e1/stops-claude.jsonl`, and
   `artifacts/usage/ledger-claude.jsonl` (tokens and `plan_equivalent_usd`, an API-price
   estimate, never cash). These files are written only on Raj's machine, so they merge with
   this session's work without conflicts; push to the session branch or send the files back.
