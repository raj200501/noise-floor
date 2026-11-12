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
85-task E1 shape (D-034 item 1): about 8 core-minutes per dataset in a timing run at 80 tasks,
1,200 GLMM datasets (G6 200, E0 groups `e1` and `e1-het` 500 each), so a little over 40 hours on
4 workers, plus the cheap local-arm and non-GLMM cells). Both studies write all their rows before
either is banked; then bank both, run `python -m noisefloor.e0_validity`, and commit its outputs
(the list of failing cells under `src/evalstats/data/`) before the local arm's first batch and
before Raj clones for the Claude slice; then the E1 local arm (both arms refuse to start before
the G6 and E0 part 2 artifacts are banked). Running G6 and E0 part 2 first lets a failing kind of
interval be replaced by a dated amendment with its own passing cells before any E1 data exist.
No commit changes `src/` or `uv.lock` from the start of G6 to the E1 analysis, except the one
`e0_validity` commit above, which lands after G6 and E0 part 2 have written all their rows (the
list of failing cells is outside the analysis-code pin): the studies refuse rows from a second
environment, the arms stop on new run code, and the analysis runs only under the analysis code
G6 and E0 part 2 ran (D-033 item 10, D-034 item 8). The pinned files include `noisefloor.e1`
and `noisefloor.models`, so they stay unchanged through G7's recompute in the final ship-check;
a repair to one of them during an arm, which a resume accepts, also needs a frozen amendment of
0004 with a ```json analysis-code``` block before the analysis runs. Both arms check before they
start that the analysis code at HEAD is the code G6 and E0 part 2 recorded and that the list of
failing cells is rebuilt from the banked coverage (D-034 item 16). Raw rows are committed and pushed at
check-ins (while E1 runs, pushes wait for the gaps between blocks), because this container is
not persistent: only pushed rows survive a new one. Runs banked after the last push and lost with
a container are noted in the stop log (`python -m noisefloor.e1 note local --reason "..."`) and
run again under their planned run ids, a deviation. The llama-server build lives in
`~/.cache/noisefloor`, which a new container does not keep: a rebuilt server whose digest
differs from the frozen pin starts the local arm only under a frozen amendment of 0004 with a
```json llama-server``` block naming the new digest, a deviation (D-034 item 9). Nothing else runs on the CPU while E1 runs,
because its runs have wall-clock limits.

- The local arm runs as `python -m noisefloor.e1 local prereg/0004-e1-factorial.md`, detached
  (`setsid nohup`, PID file and log under `runs/e1/`), with a watcher. It is resumable: a
  restart banks completed runs from the work directories and starts unfinished ones again.
- The first batch starts within 15 minutes after 00:00 UTC (launched later, the runner waits).
  A batch of slot day k, block b starts only in block b of the arm's day k or later, so the arm
  takes at least four UTC days (8 slots of 9 to 10 hours at the dry run's rates, `e1-design`
  `duration_estimate`); slots that overrun move to the same block of the next day. The dry run
  had no violation-family runs (the slowest family in screening), so slots may run about 10.5
  to 12 hours, leaving gaps between blocks of about an hour or none (D-038): plan pushes for
  short gaps, and after day 1 re-forecast from the timestamps in `runs/e1/alive.txt` (counts
  and times only, never outcomes).
- Banked runs are committed after each check-in and pushed (after `make prepush`, which needs
  the CPU) at the next gap between blocks; pushes happen only while the arm waits between
  blocks. The same holds for tests,
  simulations, and anything else CPU-heavy in this session: CPU contention would lengthen runs
  and could turn some into timeouts, an effect outside the design.
- Check-ins keep the session awake while the arm runs; each one checks the process, the log,
  and the banked count, and relaunches the runner if it died. The log names banked and
  infra_error counts and infra_error end reasons only, never outcomes, and
  `artifacts/e1/local.jsonl` is read only by the runner until the arm ends.
- One runner at a time: each arm holds an exclusive lock (`artifacts/e1/.local.lock`,
  git-ignored); a second runner exits with code 4. Each batch runner holds a lock on its work
  directory, and no batch starts while any runner under the work root holds its lock, so a
  relaunch after a killed orchestrator waits for a live runner instead of running beside it.
  The sandboxed runner gets SIGKILL when the process that launched it dies (`unshare --fork`
  blocks SIGTERM, D-034 item 10), and a runner's model server stops when the runner dies.
- Stops (prereg 0004): a batch that fails as a whole (the model file fails its hash check, the
  server does not start or fails its health check after a run, three infra_error runs in a row,
  or the runner exits before every run is recorded) banks only the runs the failure did not
  touch; the others run again later under their planned run ids. The arm then stops, as it
  does when more than 5 percent (and at least 5) of the runs banked since its start or last
  resume are infra_error, and when a batch would start after day 7. Each stop is a line in
  `artifacts/e1/stops-local.jsonl` (reported as a deviation), and the runner exits with code 3
  until `python -m noisefloor.e1 resume local --note "<repair>"` records the repair. A batch
  that banks nothing because its calls were refused, or new run code (a commit under `src/`, a
  new lock, or another llama-server build, its binary or shared libraries), also stops the arm.
  While `src/`, `uv.lock`, or `pyproject.toml` has uncommitted changes (staged edits and
  untracked files included), the runner logs "commit first" at its next batch boundary and
  exits without a stop row; do not commit (that records a code change and needs a resume, a
  deviation): run `git restore --staged --worktree -- src uv.lock pyproject.toml`, list
  untracked files with `git clean -n -- src` and remove them, then relaunch (D-038;
  `tools/runcheck.py e1-relaunch` refuses a dirty tree). A defect found during the arm is recorded
  with `python -m noisefloor.e1 stop local --reason "..."` (it takes effect when the running
  batch ends) and resumes only with `--amendment` naming a frozen amendment of prereg 0004 that
  says how the affected runs are handled and names the stop's time in a ```json defect-resume```
  block. The day-7 stop does not resume.
- Operating rules (D-036). G6 and E0 part 2 are relaunched only through
  `tools/e0g6_driver.sh` (a preflight first; it stops at the first step that fails) and watched
  by `tools/wait_e0g6.sh`. The local arm launches only if the fast ship-check at the banking
  commit prints G5 PASS and G6 PASS. It is launched between 21:00 and 23:45 UTC from a fresh
  shell, after every commit: `mkdir -p runs/e1 && setsid nohup .venv/bin/python -m
  noisefloor.e1 local prereg/0004-e1-factorial.md >> runs/e1/log.txt 2>&1 < /dev/null & echo
  $! > runs/e1/pid`. Then `bash tools/wait_e1.sh` runs as a harness-tracked background task,
  restarted first on every wake-up, with its two-hour limit kept off 23:45 to 00:30 UTC on
  launch night, and an hourly check-in fires into the session. Nothing in git changes until the
  first batch is banked. During batches a check-in only reads and commits; CPU-heavy work waits
  until `python3 tools/runcheck.py e1-gap --need <seconds>` prints `E1-GAP: OPEN`, and every
  shell command starts with `echo 19 > /proc/self/autogroup; echo 1000 >
  /proc/self/oom_score_adj`. After a restart or a stop the relaunch comes first: `python3
  tools/runcheck.py e1-relaunch --rotate`, a note for any `aborted.json` it lists, the same
  launch command, and `python -m noisefloor.e1 note local --reason "..."` for the interruption.
- The E1 analysis runs only after both arms have ended (prereg 0004), that is, after Raj's end
  point for the Claude slice is in this repository; the local arm alone yields no E1 result.

## Claude arm on Raj's machine (G7)

The Claude tiers of E1 run through headless Claude Code on a machine where the billing guard is
clean (D-004, D-019). This hosted session cannot run them by design. As prereg 0004 says, the
E1 analysis waits until both arms have ended, so the analysis (which also writes E3 on E1), the
post-E1 check, the report's E1 sections, and the final review all wait for this slice's end
point (D-036 item 2). Steps for Raj:

1. Start only once STATUS.md says G5 PASS and G6 PASS at the banking commit and the local arm
   has started: a confirmatory Claude run before a failing validation is resolved closes the
   preregistered route for replacing the failing method (D-036 item 1). Clone the branch
   `claude/gifted-darwin-k7q99b` (one clone only) at or after that commit, which runs
   `e0_validity` (STATUS.md names it), and check that `artifacts/g6/recovery.json` and
   `artifacts/e0/coverage-0005.json` exist. The slice needs
   macOS or Linux (on Windows, WSL). Then `make setup` and `make doctor`. Its `billing guard`
   row must read `clean`: no `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`,
   `CLAUDE_CODE_USE_*`, or `apiKeyHelper` in any user, project, or managed settings file. The
   doctor does not read settings `env` blocks, so also check that `env | grep '^ANTHROPIC_'`
   prints nothing and that no settings file has an `env` block with `ANTHROPIC_*` or
   `CLAUDE_CODE_USE_*` entries (they could remap the model aliases). The `claude` CLI must be
   logged in to the Max plan (check with `/status` inside `claude`). Turn off paid extra usage
   in the plan settings, and turn off auto-update for interactive use too
   (`DISABLE_AUTOUPDATER=1` in the shell profile or in the `env` block of
   `~/.claude/settings.json`) until the end point: each new CLI version stops the slice until a
   smoke run on it passes.
2. Smoke check (one exploratory run that counts as one of that UTC day's `agent_run`s, so run it
   the day before the first slice day, or expect one run fewer that day):
   `uv run python -m noisefloor.claude_arm smoke --tier claude-haiku`. It exits 0 only when its
   last line is `SMOKE: PASS`: the child reported the plan login (`login: OK 'none'`, with the
   quotes), only the file tools Read, Glob, Grep, Write, and Edit, no MCP servers, no memory
   files, and a model ID (a Haiku model) and CLI version. `SMOKE: CHECK <what>` or
   `SMOKE: NOT RUN ...` (for example a login that is not the plan's, which the stream refuses)
   exits 1: stop and report it. The smoke writes `artifacts/pilot/claude-smoke.jsonl`, the
   record that the slice started after a pass.
3. Run the slice with `uv run python -m noisefloor.e1 claude prereg/0004-e1-factorial.md` from
   that clone once in every UTC day (a UTC-timed cron or launchd job works: the cap counts per
   UTC date, and a day without a call is lost). It is resumable. The runner refuses a plan,
   pins, or tasks that differ from the frozen ones, a missing SMOKE: PASS, and uncommitted
   changes under `src/` (if it says to commit first, do not: committing an accidental edit
   records a code change; run `git restore --staged --worktree -- src uv.lock pyproject.toml`,
   list untracked files with `git clean -n -- src`, remove them, and run the command again;
   for backups, stage the slice files by path, never `git add -A` or `git commit -a`). Do not merge
   or pull the session branch into the slice clone before the end point: this session pushes
   rows to it, a merged change under `src/` stops the slice until a resume accepts it as a
   deviation, and a change to a pinned file (for example `src/noisefloor/e1.py`) makes every
   later call refuse, the one that records the end included. Push mid-slice backups to a
   separate branch (for example `claude-slice`), which this session neither merges nor reads
   before the end. Daily caps are the budget.toml caps times the smaller of budget.toml's
   `cap_share` (0.2) and `CAP_SHARE`: at most 6 runs a day (about 30 days for 180 runs). A
   faster slice needs `cap_share` raised in budget.toml by Raj (Claude never raises a cap; at
   0.5 the cap is 15 runs a day); then commit budget.toml with the ledger and keep it at the
   highest value used, because `make budget` checks every past day against the current file.
   The turn cap per run is the smaller of the planned 8 steps and `claude_agent_max_turns`
   times that share (8 at 0.2, so a raise never changes it); a run whose cap would be below its
   planned steps is refused, so do not set `CAP_SHARE` below 0.2. Read each day's output: a
   `model calls refused (...)` line that names paid extra usage or an API-key login means do
   not run again until it is fixed, because the refusal stops only that day's call.
4. The slice ends when every planned run of the tiers still served is banked, or at
   2026-12-26T04:21:42Z (45 days after the freeze of 0004 at 2026-11-11T04:21:42Z), whichever
   comes first. If it has not finished by then, run the command once at or after that instant
   (on 2026-12-26 after 04:21:42 UTC, not earlier that day): it records the end point whatever
   stop is open and prints `e1 Claude slice: ended (45 days after the freeze, stopped or not)`.
   Check that the last row of `artifacts/e1/stops-claude.jsonl` has kind `end_point`. An early
   end is `uv run python -m noisefloor.e1 end claude --reason "..."`, a deviation. Run the
   smoke check on the version the slice will use: a first run on a version no smoke run passed
   on, or a later new version, stops the slice until a smoke run on it prints SMOKE: PASS and
   `uv run python -m noisefloor.e1 resume claude --note "<version change>"` accepts it as a
   deviation. If a tier is served by a model ID other than its first run's, that tier starts no
   further run and the others go on (early in the slice that can leave fewer than the 150
   Claude-tier runs G7 needs). A Claude Code child that did not start clean, three infra_error
   runs in a row (in banking order, also across a pair), or an infra_error share over the
   limit stop the slice for repair. Exit code 3 means a stop is open or the slice has ended:
   read the last row of `artifacts/e1/stops-claude.jsonl` (`end_point` means done); the resume
   command is printed on the next call. Stops, resumes, and ends of the slice are recorded only
   on this machine (the commands refuse where the billing guard is not clean).
5. After the end point, commit `artifacts/e1/claude.jsonl`, `artifacts/e1/stops-claude.jsonl`,
   `artifacts/usage/ledger-claude.jsonl` (tokens and `plan_equivalent_usd`, an API-price
   estimate, never cash), and `artifacts/pilot/claude-smoke.jsonl`, plus budget.toml if
   `cap_share` was raised. These files are written only on Raj's machine, so they merge with
   this session's work without conflicts: fetch, merge the session branch, push, and tell this
   session. The E1 analysis and everything after it start then.
