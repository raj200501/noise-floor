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

## D-009 2026-10-23 GLMM backends and variance-component intervals
- Primary backend: lme4 (`glmer`, Laplace approximation, bobyqa) through rpy2. rpy2 is pinned to
  `>=3.5.17,<3.6` as the optional extra `r`, because rpy2 3.6 needs R 4.5 and this container has
  R 4.3; `make setup` adds the extra when `Rscript` is on the PATH.
- Fallback when R is absent: statsmodels `BinomialBayesMixedGLM` (variational Bayes, extra
  `glmm`). Its intervals are approximate posterior intervals and are labeled `vb_posterior`;
  they are never used for a headline interval unless E0 shows they cover.
- Variance partition on the latent logit scale (Nakagawa and Schielzeth style): random-effect
  variances, the variance of each fixed factor's contribution to the linear predictor over the
  observed design, and the logistic residual variance pi^2/3.
- Intervals: profile likelihood from lme4 (`confint(method = "profile")`) for random-effect SDs
  and fixed coefficients, mapped to variances. For a two-level fixed factor the contribution is
  `p(1 - p) beta^2`, a monotone function of |beta|, so its interval is the image of the profile
  interval for beta. Factors with more levels get no interval from this path (reported as such),
  and the E1 design uses two-level factors. No Wald intervals for variance components.

## D-010 2026-10-23 Docker-free agentic suite for E1 (nf-tools)
No Docker daemon here, so the Terminal-Bench 2.0 arm through Harbor is skipped (recorded before
any data). The Docker-free suite is `nf-tools`, procedurally generated tool-use tasks over a mock
in-memory file system with mock tools (list, read, write, search, calculator, submit) and
deterministic graders. Reasons: it runs anywhere with no network, no code execution, and no
third-party task code; tasks are generated fresh from seeds, so no training-data contamination;
difficulty has explicit knobs, so the pilot can place every model configuration between 0.2
and 0.8 mean success (section 4); and grading is exact. Cost: less realistic than terminal
tasks; the report says so. Pilot and confirmatory tasks come from disjoint seed ranges fixed in
the preregistration.

## D-011 2026-10-23 Local subjects and harness factor
Local open-weight models (GGUF, 4-bit, pinned by repository revision and file SHA-256) are
served by llama-cpp-python's OpenAI-compatible server (CPU build). Both local harnesses talk to
the same server, so the harness factor changes only the scaffold: minimal-agent (this project's
scaffold, JSON tool calls) versus Inspect's basic agent (inspect_ai, OpenAI-compatible provider).
The Claude Code native harness cell is empty for local models and the Claude arm cannot run here
(D-004); the design stays valid with those cells empty, and any contrast that is not crossed is
labeled scaffold-confounded.

## D-012 2026-10-23 Sandbox for subject runs
Each run batch executes in a child process inside new user, network, and mount namespaces
(`unshare`): no external network (only loopback, for the local model server started inside the
same namespace), the repository mounted read-only, writes only to a temporary workspace, and an
environment with tokens, API keys, `SSH_AUTH_SOCK`, and cloud credentials removed, plus
`GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null`. nf-tools runs no model-generated code
(its tools are in-process mocks), so the sandbox mainly confines third-party harness code
(inspect_ai) and the model server. The sandbox is verified by tests before any data is collected.

## D-013 2026-10-23 Integrity tools (ported from raise-v1-public, Apache-2.0)
`tools/prereg.py`, `tools/claimcheck.py`, `tools/freshness.py` are adapted from raise-v1-public at
commit 9330c9d (git clone; see THIRD_PARTY_NOTICES.md). Interpretations, all stricter than the
brief's minimum:
- claimcheck checks untagged, `text`, and `console` code fences and digit-only inline code (only
  fences tagged with a programming or config language are skipped); result numbers must carry a
  `<!-- claim:<id> -->` marker resolving to `artifacts/claims.json` (artifact path, SHA-256, JSON
  pointer, optional derivation and format). Exemptions in docs/claims_exempt.toml are exact strings
  in context, scoped to named documents, each with a category from the brief's non-result list
  and a reason; an unused exemption fails.
- Preregistration numbers equal their chain position; amendments need `amends: NNNN` and `date:`
  front matter. A commit touches a file only if it differs from every parent. Both the prereg file
  and its chain entry must strictly predate every commit that introduced a confirmatory artifact;
  `"confirmatory": true` without a prereg fails. The brief and its hash file must each have exactly
  one committed version. prereg-verify needs full history (it fails on shallow clones).
- freshness re-derives every registered claim, used or not.
- Limits recorded in docs/INTEGRITY.md: numbers in words, in SVG figures, in HTML attributes, or in
  language-tagged code are not seen by claimcheck; results are kept out of those places.

## D-014 2026-10-23 Commit authorship (Raj's instruction)
Every commit from here on is authored as Raj Kashikar
<65059626+raj200501@users.noreply.github.com> (repo-local git config; `--author` if the
environment sets GIT_AUTHOR_*), keeping the Co-Authored-By and session trailers. Existing
commits are not rewritten.

## D-015 2026-10-23 GitHub Actions budget (Raj's instruction)
ci.yml runs only on pushes to main, pull requests into main, and workflow_dispatch (never on
claude/ branches or a schedule), with a cancel-in-progress concurrency group and
timeout-minutes 10; jobs stay lean (setup, make ci). No other workflows. `make ci` must pass
locally on a fresh clone before every push. (The stop-now commit that introduced this was pushed
without running make ci, on Raj's instruction.)

## D-016 2026-11-04 E1 pilot (exploratory)
The pilot (noisefloor.pilot, plan p1) crosses the four candidate local models, both local
harnesses (minimal-agent, Inspect basic agent), the five nf-tools families, and difficulties
1 to 5, one task seed each, at temperature 0, a 300 s wall-clock limit, 15 steps, 4 threads,
concurrency 1 (200 runs). Its task seeds come from 900000 to 900999; the confirmatory E1 seed
range will be disjoint and fixed in the E1 preregistration, and pilot data is never pooled with
confirmatory data. Pilot records go to artifacts/pilot/p1.jsonl with phase "pilot" and are
reported as exploratory. Purpose: choose models and a difficulty mix so that every model
configuration's mean success lies in [0.2, 0.8], measure run times to set the timeout levels,
and estimate variance inputs for the E1 power analysis. Batches run as split plots (one server
configuration per batch), so the E1 analysis will include a batch random effect.

## D-017 2026-11-04 E0 grid in two parts; bootstrap interval types
- E0 is frozen in two parts: prereg 0003 (score, paired, CR1, rank, and bounded-score cells,
  including cells shaped like the two E2 boards) now, and a dated amendment of 0003 (the logit
  GLMM variance-component cells, shaped like the E1 design) once the pilot fixes that design and
  before E0 part 2 or any E1 data. `artifacts/e0/coverage.json` cites 0003 as `prereg` and the
  amendment(s) under `amendments`; ship-check G5 checks the union of the base grid and its
  amendments' grids, and that each of those files is committed strictly before the artifact.
- The G5 rule allows 500 replicates "for bootstrap or GLMM intervals". BCa and the rank
  intervals (task bootstrap of the maximum studentized deviation) are bootstrap intervals, so
  their interval types (`bca`, `bootstrap_max_t`) get the 500-replicate floor, like `bootstrap`
  and the GLMM profile type.

## D-018 2026-11-04 Pilot p1 stopped early; second-round candidates; context window
- Pilot p1 (D-016) was stopped after 24 banked runs, all on Qwen2.5-1.5B: 2 of them solved their
  task, most ended at the 15-step limit (for example calling `list_dir` on a file path again and
  again despite the `not a directory` observation), and one ended when the conversation exceeded
  the server's 8,192-token context. That puts Qwen2.5-1.5B well under the 0.2 floor of section 4,
  and the two smaller first-round candidates are unlikely to do better. The 24 runs stay in
  artifacts/pilot/p1.jsonl as exploratory data.
- Context: llama-server now gives every parallel slot 32,768 tokens (fifteen steps of
  observations and replies can exceed 8,192). Earlier pilot runs used 8,192.
- Second-round candidates (all permissive licenses, pinned by revision and file SHA-256):
  Qwen3-1.7B (Q8_0, the only official GGUF), Qwen3-4B (Q4_K_M), Granite 4.0 micro (3B, Q4_K_M),
  SmolLM3-3B (Q4_K_M, ggml-org build), and Phi-4-mini-instruct (3.8B, Q4_K_M, unsloth build;
  MIT). Qwen3's thinking mode is turned off through the chat template
  (`enable_thinking: false`), because thinking tokens on CPU would multiply run times; this is a
  fixed part of each model configuration, recorded in provenance.
- A screening plan (s2) runs each candidate on a small task set with both harnesses to measure
  success and run time; the E1 models and difficulty mix are chosen from it and the pilot, then
  preregistered. All screening data is exploratory.

## D-019 2026-11-04 E1 Claude arm (runs only where the billing guard is clean)
- Two harnesses for Claude tiers: Claude Code native (`claude_code`) and minimal-agent with one
  headless `claude -p --restricted --tools ""` call per step (`minimal`), so model is crossed
  with scaffold for at least one Claude tier.
- Claude Code native: the task's mock files are written into a disposable workspace (a mock path
  `/etc/x` becomes `./etc/x`), Claude Code works on them with its file tools only
  (`--tools Read,Glob,Grep,Write,Edit`, no Bash, no web tools), `--restricted
  --strict-mcp-config --permission-mode dontAsk --no-session-persistence`, and writes its answer
  to `ANSWER.txt`; grading reads that file and the final workspace files with the suite's
  grader. Turns are counted from the stream-json output and the child is stopped at the cap.
- Caps: each run is one `agent_run` ledger entry against `claude_agent_runs_per_day`; minimal-
  agent steps are logged as `agent_step` entries (guarded, not separately capped). With
  `cap_share = 0.2` the per-run turn cap is 8 (D-007), so the Claude arm's step budget (8)
  differs from the local arms' (15); contrasts across arms are labeled step-budget-confounded.
- The arm needs a machine where `make doctor` shows the billing guard clean and the `claude` CLI
  logged in to the Max plan; in this hosted session it is refused by design (D-004). The
  instructions for Raj go in docs/PLAN.md with the E1 preregistration.
- Implementation details (src/noisefloor/claude_arm.py, client.py), settled before any Claude run:
  - Turn cap: the CLI's `--max-turns` option (present but hidden from `--help` in 2.1.318) is
    passed, and the stream reader also counts distinct assistant messages and stops the child
    past the cap, as section 8 asks when no max-turns option can be relied on.
  - $0 checks while the child runs: the init event's `apiKeySource` must be `none` (the plan
    login; `/login managed key` or an API key would bill API credits), and a `rate_limit_event`
    with status `rejected` or with paid extra usage in use (`isUsingOverage`, `overageInUse`)
    stops the child at once and stops the plan (`UsageLimitError`, reset time kept in
    `refused.json`). Refused runs are not banked and run again later.
  - Clean child: `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` keep
    Raj's CLAUDE.md files and auto memory out; parent-session markers and other services'
    credentials are removed; the init event's model, tools, MCP servers, and memory paths are
    recorded per run (`claude.clean`). `python -m noisefloor.claude_arm smoke` runs one pilot
    task and prints this check; Raj runs it once before the arm starts.
  - Where it runs: the Claude Code process needs the network to reach the model, so Claude
    batches run as a host process, not in the network-less namespace sandbox. The subject has no
    tool that runs code or reaches the network, and `--restricted` confines its file tools to a
    temporary workspace outside the repository; minimal-agent tools are in-process mock tools.
  - Answer: `ANSWER.txt` as it stands when the child ends; writing it counts as submitting.
  - Minimal-agent with Claude: each step is one stateless call with the conversation so far as
    one role-tagged prompt, because the CLI cannot replay earlier assistant turns. Temperature
    and seed cannot be set through the CLI; they stay in the run spec and the cache key only.

## D-020 2026-11-04 SWE-bench reader tie-breaks; screening s2 stopped for suite fixes
- `read_swebench` now breaks ties between lists of equal status by a fixed table order (a
  specific list, then `applied`, then `with_logs`, then `generated`; other ties by the order of
  `RUN_REPORT_LISTS` or `RESULTS_CATEGORIES`), not by the file's JSON key order, and treats only
  an absent or null list as empty. This changes only `status_reason`; no banked artifact uses
  this reader. Open limitations, kept as they are and listed in the adapter's tests: patch-apply
  failures and missing predictions in a run report map to `infra_error`, and swebench's own
  "ambiguous" tier other than timeouts maps to `ok`.
- An adversarial review of the nf-tools suite and both harnesses (before the E1 preregistration)
  found defects that would bias E1 statuses and grading (D-021). The s2 screening was stopped
  after 11 runs (banked in `artifacts/pilot/s2.jsonl`, exploratory) so that screening and E1 use
  one suite version; a new screening plan runs on the fixed suite.

## D-021 2026-11-04 nf-tools suite v2 and harness parity (before any E1 data)
An adversarial review (D-020) found defects that would bias E1. Fixed before the E1
preregistration; suite version 2 (new golden manifest hash). Pilot p1 and screening s2 ran on
version 1: exploratory, never pooled with version 2 data.
- Grading: answers go through one extraction rule in every harness (`extract_answer`): code
  fences, a one-field JSON object, wrapping quotes, curly quotes, backticks, or markdown emphasis,
  a leading `The answer is` or `Answer:`, a one-word label with `=` or `:` (`port = X`, `id=X`),
  a trailing note in parentheses, trailing `. , ; : !`, and for numbers a currency sign or a
  trailing unit word. No expected answer contains any of these, so extraction can only accept a
  correct value. The strict comparison is kept as `outcome_strict` for a sensitivity analysis.
- Instructions: config asks for "only the value, exactly as written in the file, not the key
  name"; derive adds "Do not write a header line".
- End reasons in both local harnesses: a wall-clock expiry during a model call is `wall_clock`
  (status `timeout`), a context overflow is the new `context_limit` (status `timeout`: a limit of
  the run, caused by the episode), and other server failures stay `backend_error`
  (`infra_error`). The proxy returns llama-server's context error to Inspect as OpenAI's
  `context_length_exceeded`. Regression tests run both harnesses on a scripted server.
- Inspect: only submit calls that Inspect accepted count. With `max_attempts=1`, a submit that
  fails Inspect's argument validation (for example a number) ends the sample with no answer
  (`stopped`, `submit_rejected`); minimal-agent accepts numbers. This is a property of the
  harness and stays part of the harness contrast.
- Shared counts: `tool_calls` are calls that reached the episode's tools, without `submit`;
  `rejected_calls` are actions a harness rejected before any tool ran (minimal-agent: parse
  errors and unknown tools; Inspect: validation errors).
- minimal-agent parser: an action that names a known tool wins over other JSON in the reply;
  arguments may sit under `args`, `arguments`, `parameters`, or `input`, as a JSON string, in
  the OpenAI function wrapper, or flat beside the tool name.
- Tools: `read_file` and `write_file` reject a path ending in `/`; path arguments lose
  surrounding whitespace, backticks, and quotes; error messages name the tool to use; `calc`
  limits nesting, not the length of a flat sum; `ToolEnv.call` never raises.
- pointer: `old:` lines go on every note (difficulty 3 to 5) and `previous code:` on every note
  that holds a code (difficulty 5), so no line type marks the chain or its end. A search for
  `next:` still lists every link; tracing the chain from that listing is legitimate and costs
  reasoning rather than tool calls.
- derive: each distractor table holds at least one new sku of the asked category that passes the
  threshold, so reading it by mistake changes the output; the archive copy and the `.bak` are
  drawn independently. Derive episodes are graded on the final file whatever the end reason; E1
  will report how many timeout rows have outcome 1.
- csv_sum difficulty 4 and 5 are unchanged: they fit the 15-call budget only with search-based
  discovery (the test oracle now runs under that budget), which the E1 preregistration states.

## D-022 2026-11-04 E1 decomposition method and design, from exploratory simulation pilots
All evidence here is exploratory (method pilots on seeds 990000-990359, synthetic data only;
never pooled with the confirmatory G6 or E0 seeds).
- Profile-likelihood intervals are not usable for E1: one E1-sized fit took 8 minutes and every
  interval failed. Intervals use the parametric bootstrap instead (allowed by the brief), with
  `nAGQ = 0` fits and refits: point estimates matched Laplace closely (model:task variance 0.204
  against 0.197 and 0.422 against 0.417 on the same data) at about a tenth of the time.
- No batch random effect in the primary decomposition (supersedes D-018's plan): its variance was
  estimated about 40 percent low and its intervals covered at most 0.75 (pilot 1). Batches are
  randomized over slots, so whole-plot factors stay in the model; a decomposition with a batch
  term is reported as an exploratory sensitivity analysis.
- Day and time-of-day block enter as fixed factors: with 3 days, or 2 blocks, a random-effect
  variance cannot be estimated with valid intervals (a likelihood-ratio interval for 6 groups
  covers about 0.906, and less for fewer).
- 16 runs per (model, task) cell (the full half fraction): with 8 binary runs per cell the
  model:task variance was estimated about 20 percent low and no bootstrap rule reached nominal
  coverage (pilot 3, 60 datasets: percentile 0.72, basic 0.77, log-basic 0.77, bias-corrected
  0.85); with 16 the point bias fell to about 5 percent (20-seed scans).
- The model-by-harness question is answered by per-model paired harness contrasts (task-clustered
  t intervals), not by an interaction term in the decomposition, whose variance split would
  depend on the contrast coding.
- The interval rule for each kind of component, and the MDE interval, are chosen from pilot 4
  (the candidate E1 shape) before the E1, E0 part 2, and G6 preregistrations freeze them.

## D-023 2026-11-04 Laplace bootstrap for E1, and how its interval rules are chosen
Supersedes the nAGQ = 0 choice and the "chosen from pilot 4" step of D-022. Evidence is
exploratory and banked in `artifacts/method_pilots/` (synthetic data, seeds from
`PILOT_SEEDS`, never pooled with G6 or E0); `python -m noisefloor.method_pilots` recomputes
`summary.json` from the raw rows.
- Why: in pilots 4 and 5 (nAGQ = 0, the candidate E1 shape) the contribution boxes of the
  two-level factors covered below nominal under every box rule, with harness lowest, and the
  empirical spread of each coefficient exceeded the spread its intervals implied
  (`pools.nagq0-e1-shape`: `components.<factor>.coverage` and `sd_ratio`). nAGQ = 0 fixed
  effects were attenuated against Laplace on the same simulated tables, which narrows a
  bootstrap built from them. The studentized box (pilot 5) did not help.
- Change: the original fit and every bootstrap replicate use the Laplace approximation
  (`nAGQ = 1`). Replicates are fresh `glmer` fits that skip the final derivative pass (lme4's
  `refit` was slower for Laplace); convergence diagnostics come from the original fit.
- Shares of the total get their own interval rule (`share_interval`), and G6 and E0 part 2
  score every component's share, the residual's included, like the components. Candidates:
  the four variance rules on replicate shares, and `plugin`, each component's own interval
  (under the rule chosen for its kind) mapped through `v / (v + rest)` with the rest of the
  total at its estimate, the residual share taking its percentile interval. Added before pilot
  6 was read because a planted share of exactly 0 (a null factor) can never sit inside a
  percentile, bias-corrected, or log-scale interval of positive replicate shares; under
  `plugin` a contribution box that reaches 0 gives a share interval that reaches 0. Shares are
  chosen after the random and fixed rules.
- Pilot 6 (exploratory, nAGQ = 1, seeds 990500 to 990599, the pilot 4 design) scores every
  rule. The rule for each kind of result is fixed here before its results are read: random
  variances (task, model:task), fixed-factor contributions (every fixed factor), shares (every
  share), and MDE rows (the three pilot designs) each get the rule with the highest minimum
  coverage over that kind's components in pilot 6; rules within 0.01 of that minimum are
  broken by the smaller median interval width, summed over the kind's components with each
  component's width divided by its median planted value (1 when that is 0)
  (`noisefloor.method_pilots.choose_rule`, written before pilot 6 was read). If the
  best minimum for a kind is below 0.90, that kind's intervals do not become headline
  intervals (the E1 preregistration then reports them as estimates with unvalidated intervals),
  and the failure is recorded here.
- Score intervals use `cluster_wilson` (Wilson at the design-effect sample size): in an
  exploratory check the task-clustered t interval under-covered at high success rates with 2
  runs per task, the Claude slice's shape. E0 part 2 adds score cells across success levels
  for both arms, and a paired cell with 1 run against 4 for the Claude-versus-local
  comparisons.

## D-024 2026-11-04 CPU priority on this machine: autogroups, and s3's timing caveat
- Finding: the kernel's scheduler autogroups are on (`/proc/sys/kernel/sched_autogroup_enabled`
  is 1), and every detached job (`setsid`) gets its own autogroup, inside which `nice` applies;
  between autogroups `nice` has no effect. So the exploratory method pilots (launched with
  `nice`), timing checks, and test runs competed with the screening runs at equal weight. At
  08:50 UTC llama-server held about half the CPU it needs while two pilot processes ran.
- Consequence: screening s3 (exploratory) ran under varying CPU contention from 07:11 UTC on,
  so its wall-clock times are upper bounds and some of its timeouts may come from contention;
  phi-4-mini's first batch ran under the heaviest load. s3 serves only to choose E1's models
  by success rate, and its records are banked as they are, with this caveat.
- Rule from now on: a CPU job that runs beside model runs sets its autogroup to the lowest
  priority (`echo 19 > /proc/<pid>/autogroup`) as soon as it starts; with that, llama-server
  got its three threads back at once. While the E1 local arm runs, nothing CPU-heavy runs at all
  (docs/PLAN.md), whatever its priority.

## D-025 2026-11-04 E1 local models from screening s3, and s3 stopped early
- Rule (s3 is exploratory, D-021 suite version 2): an E1 local model needs an s3 success rate
  from 0.2 to 0.8 (the brief's pilot target) and runs that can finish inside E1's shorter
  wall-clock limit of 150 s, so its s3 median wall time must be under 150 s. Numbers:
  `python -m noisefloor.pilot s3 --summary` on `artifacts/pilot/s3.jsonl` (with the D-024
  contention caveat).
- Chosen: qwen3-1.7b and qwen3-4b (Qwen3) and granite-4.0-micro (Granite 4.0): three
  configurations over two families, as the brief requires (section 5). Excluded: smollm3-3b
  (success below 0.2, most runs timed out) and phi-4-mini.
- phi-4-mini: s3 was stopped at 08:55 UTC after phi's first 4 runs, all failures at a limit
  (three at the 300 s wall clock, one at the step limit). Its llama-server generated about 5
  tokens per second even once the CPU was free (granite-4.0-micro about 11), so a 15-step
  episode cannot finish in 150 s and would add about a day of CPU to E1 for runs that mostly
  time out. Its 4 completed runs were banked by the normal salvage path (`salvaged: true`);
  the run in flight at the stop was discarded. The other 16 planned phi runs were not run.
- With 3 local models the local arm is 3 x 30 tasks x 16 runs = 1,440 runs; with the Claude
  slice's 180, E1 plans 1,620 runs (G7 needs at least 1,000, at least 150 on Claude tiers).

## D-026 2026-11-04 Each E0 grid banks its own coverage artifact; G5 reads the chain
- Problem (prereg review, before E0 part 2 is frozen): `e0_run bank` wrote every grid into
  `artifacts/e0/coverage.json`, whose first commit is the part 1 bank, so listing amendment 0005
  there could never satisfy "the preregistration predates the artifact" (G5 and prereg-verify).
  And G5 took the grids to check, and the headline cells, from that artifact's own fields, so an
  artifact that left 0005 out would have skipped the E1 cells entirely.
- Change: each grid gets its own artifact citing its own preregistration
  (`coverage.json` for 0003, `coverage-NNNN.json` for each amendment), so every amendment's data
  is first committed after it. G5 now finds the grids in `prereg/CHAIN.jsonl` (every frozen file
  with an e0-grid block) and the headline cells in the frozen texts (0003's list and each
  amendment's headline block), and requires every one of them. D-017's rule is unchanged.

## D-027 2026-11-04 E1 day and block per batch; the arm starts in block 1; simulator batches
From the prereg review, before freezing E1:
- Day and block enter the E1 analysis per batch: the UTC date and time-of-day block of the
  batch's start (`batch_started_at`), so all runs of a batch share them as the plan and the
  simulator assume (a run that starts after 12:00 or 24:00 in a batch begun before stays in the
  batch's block). A salvaged batch keeps the start it recorded when it began (`.started` in its
  workspace), not the salvage time.
- The local arm's first batch starts in block 1 (00:00 to 12:00 UTC) and day 1 is the date it
  starts; with day gating (D-025 follow-up in `eligible_batches`) each slot then runs on its
  planned day unless a slot overruns its 12-hour block.
- The recovery simulator now deals each (model, concurrency) group's batches over the six slots
  round robin from a random offset, as `noisefloor.e1.local_plan` does, instead of one global
  permutation. Method pilots 1 to 6 ran with the earlier assignment (exploratory; noted in
  `artifacts/method_pilots/manifest.json`).

## D-028 2026-11-04 Response to the E1 preregistration review (before freezing)
Five reviewers and their verifiers read the drafts of 0004 to 0006 (workflow run
wf_c2d326f8-d6b). Gate and bookkeeping fixes are D-026 and D-027 and the commits around them;
the method decisions, written before acting and before any pilot 6 coverage was read:
1. Fixed-factor contributions get task-jackknife intervals: each coefficient's interval is
   its estimate plus or minus the t quantile on T - 1 degrees of freedom times its delete-one-
   task jackknife standard error (T refits), at the Bonferroni level for a factor's k - 1
   coefficients, and the contribution's interval is its range over that box. Reason: the
   parametric bootstrap simulates from the main-effects model, so it omits task-by-factor
   variation in the effects (the harness and step-budget effects plausibly depend on the task,
   D-021), and would then make the fixed-factor intervals too narrow; a task jackknife resamples
   whole tasks and keeps that variation. Random-effect variances, their shares, and the MDE rows
   keep the parametric bootstrap (pilot 6, D-023). Exploratory pilot 7 checks the jackknife with
   and without planted task-by-factor variation before E0 part 2 validates it.
2. D-023's headline rule, tightened before pilot 6 is read (review: the 0.90 floor sat below
   G5's pass point): a kind becomes headline when, under its chosen rule, its coverage pooled
   over its components and datasets is at least 0.935 and every component's is at least 0.90.
   Otherwise its intervals are reported as unvalidated estimates, its cells stay in E0 part 2 as
   non-headline cells, and failing ones are listed as unsupported.
3. Sampling: the logistic residual pi^2/3 is the latent scale's fixed observation-level
   variance, not a measurement of run-to-run sampling, and E1's half fraction repeats no
   configuration. E1 therefore adds two replicate runs to every (model, task) cell: a second run
   of the MDE reference configuration (minimal-agent, 300 s, 15 steps, temperature 0.0,
   concurrency 1) and a second run of the fraction's configuration minimal-agent, 300 s, 15
   steps, temperature 0.7, concurrency 2. Each pair gives a within-configuration variance; their
   means estimate run-to-run sampling at the two temperatures, and the temperature 0.0 one checks
   the MDE's assumption of independent runs. The replicates stay out of the decomposition.
4. A post-E1 check at the fitted values: 200 datasets (seeds fixed in 0004) simulated from the
   E1 fit, with the batch variance of the batch-term fit and task-by-factor variances from one
   auxiliary fit, analyzed by the unchanged primary analysis. A component whose coverage fails
   the G5 rule there is reported as resting on an unsupported method. If the batch-term fit's
   batch variance exceeds 0.05, the concurrency, day, and block intervals are also reported as
   not validated.
5. Planted values are design choices: task variance 4.0 (screening s3 showed many tasks with
   identical outcomes across all runs, which variance 2.0 makes unlikely), model:task 0.5, the
   factor effects as before. E0 part 2 adds a group with planted task-by-harness, task-by-steps,
   and task-by-timeout variances of 0.25 each, analyzed by the primary analysis, for the fixed
   contributions.
6. The scaffold question gets its own test: for each pair of local models, the difference of
   their harness contrasts, paired by task (one Holm family); and the Claude tiers are compared
   pairwise under each harness (one Holm family). "Supported" means a Holm-adjusted p below
   0.05 within its family everywhere.
7. Operations: a batch that fails as a whole (server start, model check, runner exit before a
   run) banks nothing, so its runs run later in their slot's block; the arm stops when a batch
   banks nothing or when infra_error runs exceed 5 percent of banked runs; primary fits fall
   back through a fixed optimizer sequence; a bootstrap kind with more than 10 percent failed
   replicates reports no interval. The Claude slice ends when its 180 runs are banked or 45 days
   after the freeze, whichever comes first, nothing about its outcomes is read before that, its
   plan runs complete (tier, task) pairs back to back, and a run whose turn cap would be below 8
   is refused. A dry run on pilot seeds at the E1 server settings precedes the arm.

## D-029 2026-11-04 Pilot 7 and the operational stop rules (before freezing)
1. Exploratory pilot 7 (banked in `artifacts/method_pilots/`, pools `pilot7a-jackknife` and
   `pilot7b-jackknife` of `summary.json`) scored only the task-jackknife boxes of D-028 item 1,
   at the E1 shape of D-025 (3 models, 30 tasks, task variance 4.0), without (7a) and with (7b)
   planted task-by-harness, task-by-steps, and task-by-timeout variation. Under the D-028
   headline rule the fixed kind passes in both, so the jackknife is the fixed-factor rule that
   E0 part 2 validates and E1 uses; the bootstrap box of pilot 6 stays an exploratory
   alternative. Pilot 7 ran 2 bootstrap replicates per dataset to save CPU, so its bootstrap
   columns are not interval estimates and are not read.
2. D-028 item 7 in detail, as the code now does it. A batch fails as a whole when the model
   file fails its hash check, the server does not start, the server fails its health check
   after a run that did not end `ok`, three runs in a row end `infra_error`, or the runner
   exits before every run is recorded. The runs the failure did not touch are banked; the
   failing runs and the runs that never started are not banked and run again later under their
   planned run ids. Any such failure stops the arm (stricter than "a batch that banks
   nothing"). The infra_error stop counts runs banked since the arm started or last resumed and
   needs at least 5 infra_error runs as well as more than 5 percent, because a single isolated
   error early in the arm would otherwise stop it with nothing to repair. Each stop is a line in
   the arm's stop log (`artifacts/e1/stops-local.jsonl`, `stops-claude.jsonl`), reported as a
   deviation; the arm starts nothing until `python -m noisefloor.e1 resume` records the repair.
   A defect found during the arm (`python -m noisefloor.e1 stop`) resumes only under a frozen
   amendment, and the day-7 stop of the local arm does not resume. One runner per arm holds a
   lock. During the arm the log names banked and infra_error counts and infra_error end reasons
   only. Records are appended one write each, a cut final record is removed before the next
   append, and a run id banked twice fails the analysis and G7.

## D-030 2026-11-04 Claude slice: model IDs, CLI version, turn cap, files (before freezing)
Written before acting, to settle review items ops-5, ops-6, ops-17, and stats-18 together with
D-028 item 7.
1. Tiers run by their CLI alias (haiku, sonnet, opus). Every run records the model ID the CLI
   reports. A tier's first banked run fixes its model ID for the slice: a run reporting another
   ID ends the slice (it starts no further run), and runs on another ID are left out of the
   analyses and counted. Reason: mixing two models in one tier would make the tier contrasts
   uninterpretable, and stopping is the only rule that keeps every analyzed tier one model.
2. Claude Code children run with `DISABLE_AUTOUPDATER=1`, and every run records the CLI version.
   A version other than the slice's first (or one a resumed stop already accepted) stops the
   slice until a resume accepts it; the change is reported as a deviation and the runs are
   analyzed by tier. The CLI does not change the model, so this stop can resume.
3. A run whose turn cap (`claude_agent_max_turns` times the cap share) would be below its
   planned step budget is refused before it starts, and the analysis leaves out any banked run
   whose recorded turn cap differs from its `steps` factor. The smoke run uses the E1 budget of 8.
4. The plan runs complete (tier, task) groups back to back, tasks in a fixed random order and
   tiers taking turns, so a slice that ends early leaves complete pairs balanced over tiers.
5. The slice starts no run 45 days or more after the freeze (D-028 item 7), and the E1 analysis
   refuses to run until both arms have ended (the local arm: every planned run banked or the
   day-7 stop; the slice: every planned run banked, its end point, or 45 days after the freeze),
   so no outcome is read early.
6. Claude usage on Raj's machine goes to `artifacts/usage/ledger-claude.jsonl` and the slice's
   stops to `artifacts/e1/stops-claude.jsonl`; `make budget` counts both ledgers. The two
   machines then never append to the same tracked file.
7. G7 counts only runs the primary analyses use (status other than infra_error and not left out).

## D-031 2026-11-04 How the D-028 additions are built and validated (before freezing)
1. Replicates (D-028 item 3): every (model, task) cell runs its two replicate configurations a
   second time (phase `e1-rep`, run id `<planned id>-rep`, own sampling seed), batched like any
   other run, so the local arm has 1,620 runs. Replicates stay out of the decomposition, the
   scores, the contrasts, and E3; the sampling table (`sampling.json`) pairs each with its planned
   run and reports, per configuration, the mean of `(y1 - y2)**2 / 2` (the within-configuration
   outcome variance, `E[p (1 - p)]` under independent runs) with half the task-clustered Wilson
   interval of the share of discordant pairs, and, for the MDE reference configuration, the
   per-model `E[p (1 - p)]` the MDE assumes there. That interval is a score interval on 30 tasks
   with 3 values per task, so E0 part 2 validates it with score cells of that shape rather than
   through G6.
2. Heterogeneity contrasts (D-028 item 6): `evalstats.contrast_difference` gives the per-task
   difference of two models' harness contrasts with a cluster t interval; the E1 analysis and the
   G6 local intervals (`hdiff:<a>-<b>`) call the same function, so E0 part 2's local group
   validates the E1 computation. Claude tier pairs under each harness use mover_wilson, the shape
   of the P-c-w cells.
3. The Claude harness contrasts are within one tier, and each task's pair also differs in
   timeout; E0 gets a within-model paired simulator with shared task and model-by-task effects
   and that timeout split (review stats-4). E0 GLMM groups may carry planted overrides, so the
   task-by-factor group of D-028 item 5 and local-arm groups at other intercepts (review
   writing-5) run from the same recovery-config.
4. The E1 analysis now passes `fixed_interval` and `jackknife_group` to `decompose`, as the G6
   recovery does; before this change E1 would have used the bootstrap box while G6 and E0
   validated the jackknife.
5. E3 on E1 is its own output, `artifacts/e1/e3_bounds.json`, citing the E1 preregistration; G8
   reads it beside `artifacts/e3/bounds.json` (E2 boards, prereg 0001). Every E1 output names the
   tasks as procedurally generated mock tasks, and every runs row carries `mock_tasks: true`.
6. Provenance (review ops-11): every G6 row, E0 raw line, and post-E1 check row carries a stamp
   (environment hash, commit, a dirty flag for `src/`, time); the environment is the tree hash
   of `src/`, the blob hash of `uv.lock`, and the Python, NumPy, SciPy, pandas, R, lme4, Matrix,
   BLAS, and LAPACK versions. Banking refuses rows from more than one environment or a dirty
   `src/`, so commits that leave `src/` and the lock alone may land while a study runs.
7. The post-E1 check (D-028 item 4) is `noisefloor.post_check`: seeds 980000 to 980199, planted
   values derived from the banked E1 fit (with the fitted fixed effects now in
   `variance_components.json`), the batch variance of the batch-term sensitivity fit (0 if that
   fit failed, recorded), and task-by-harness, task-by-steps, and task-by-timeout variances from
   one auxiliary point fit. Its configuration is written once before the first dataset runs.

## D-032 2026-11-04 Dry run, the power analysis, labels, and planted values (before freezing)
1. E1 dry run (exploratory; `python -m noisefloor.e1 dry-run`, banked in
   `artifacts/pilot/e1-dry.jsonl` and `e1-dry-batches.jsonl`): 48 runs on pilot seeds from
   900600, every E1 model at server threads 3 and 4 and concurrency 1 and 2, 32,768 tokens per
   slot. Every server started, no batch failed, and no run ended `infra_error`; at every model
   and concurrency level the median wall time was lower at 4 threads than at 3, and the lowest
   available memory, for qwen3-4b at concurrency 2, still left several GB free. Its summary
   (`--summary`) is read for infrastructure only: server start, infra_error counts, wall times,
   load, and memory; no outcome enters any decision. E1 keeps 4 server threads and 32,768 tokens
   per slot at both concurrency levels, so concurrency does not change the context limit
   (D-018).
2. Brief section 5 asks for sample sizes from a preregistered power analysis under the section 8
   caps. No section 8 cap binds the local arm, whose runs are bounded by CPU time; its design (3
   models x 30 tasks x 16 runs plus 180 replicates) is the largest that fits the 7-day window at
   the dry run's wall times, and the caps bind only the Claude slice (6 agent runs a day at cap
   share 0.2), whose size is the G7 floor with a margin (180 runs). The preregistered power
   analysis is the `e1-power` block of 0004, generated by `tools/prereg_blocks.py`: the
   pilot-7 share of datasets in which each fixed factor is read as non-zero (with the files'
   hashes), and the normal-approximation power of each confirmatory contrast family at stated,
   assumed effects. A family or factor below 0.8 there is reported as underpowered beside its
   results, and a contrast that is not supported there is not read as evidence of no difference.
3. D-019 called the Claude-local contrasts step-budget-confounded. They compare Claude runs with
   local minimal-agent runs at the same 8-step budget, so the label is now
   temperature-confounded: Claude runs at the CLI's default temperature, the local runs at 0.0
   and 0.7 (review writing-19).
4. Planted values (review writing-15): the factor effects (harness 0.5, timeout 0.4, steps 0.4,
   temperature 0.1, concurrency 0), the day and block effects, and the model:task variance 0.5
   are the values set when the simulator was written (commit 90b4028), assumed and not estimated
   from any data; the method pilots used the same values. The task variance is 4.0 (D-028 item
   5). The post-E1 check repeats the validation at the E1 estimates.
5. The JSON blocks of 0004 to 0006 are generated by `tools/prereg_blocks.py` from code and banked
   artifacts: the interval rules and their headline status come from the pilot summary, not from
   typing; the plan hash and model pins from the code; the `validated-range` block lists the
   planted success rates the headline cells span. `noisefloor.prereg_consistency` checks that
   0005 and 0006 simulate the E1 shape with the E1 settings, and the test suite runs it on the
   frozen files.
6. Before freezing, from writing the texts: a jackknife factor whose refits more than 10 percent
   failed gets no interval (as for the bootstrap kinds); the scores table counts timeout runs and
   those with outcome 1 per model (D-021); `variance_components.json` records whether the arm kept
   the simulated schedule (3 days, every batch on its slot); and E0 part 2 adds a local-arm group
   at intercept -3.0 and Claude cells at intercept -2.0, so low scores lie inside the validated
   range.

## D-033 2026-11-04 Second review of 0004 to 0006: sample size, pins, and stop rules (before freezing)
The second review (five lenses, then adversarial verification) confirmed problems in the
drafts of 0004 to 0006. These choices answer them; none rests on E1 data, which do not exist.
1. Sample size (review brief-1, resolution-1, stats-7). D-032 item 2 said the local design was
   the largest the CPU allows in the 7-day window; that is wrong (at the dry run's rates the 1,620
   runs need about a day), and the claim is withdrawn. The preregistered power analysis now sets
   the local sample: the smallest tasks per family at which every local harness contrast and
   local model pair has power 0.8 under the Holm reading within its family, at the assumed
   effects (normal test statistics simulated with a fixed seed, the smaller power over test
   correlations 0 and 0.5). That is 16 tasks per family, 80 tasks (revised by D-034 item 1: 17
   tasks per family, 85 tasks, under the t statistics of the tests E1 runs); `tools/prereg_blocks.py`
   checks the rule when it writes the `e1-power` block. The local arm runs 3 models x 80 tasks x
   18 runs over 4 days x 2 blocks; the `e1-design` block estimates each slot's hours from the
   dry run (a planning estimate, not a result). The Claude slice keeps 30 tasks, the first 6 of
   each family, because its caps bind (180 runs at 6 a day). The planted effect of the fourth day
   is 0.05, a design value. Harness differences and every Claude family stay below 0.8 under the
   Holm reading and are reported as underpowered. G6 and E0 part 2 simulate the 80-task shape
   (one dataset took about 8 minutes on one core in a timing run, so the 1,200 GLMM datasets take
   about 40 hours on 4 workers).
2. Hypotheses (brief-2): 0004 states its tests as hypotheses with frozen decision rules and names
   the estimation targets that have no test.
3. Claude success near the ceiling (brief-7). Section 4 asks for mean success between 0.2 and
   0.8 for every model configuration. The Claude tiers could not be piloted here (D-004, D-019;
   the smoke run is one pilot task) and the suite's difficulty was set on the local models
   (D-025), so Claude success above 0.8 is expected. That is a deviation from section 4, stated
   in 0004 and reported beside the Claude results; the Claude score cells reach the ceiling.
4. Replication over days (brief-8): each (model, task) cell runs its half fraction once, dealt
   over 4 UTC days x 2 blocks, with day and block as fixed blocking factors; no day holds a full
   replicate of a cell. A full replicate on each day would multiply the local arm by four.
5. Resource limits (brief-9): the step budget is the agent's resource budget and is estimated;
   CPU and memory are held fixed (one llama-server with 4 threads on 4 CPUs, 16 GB), so E1
   estimates no CPU or memory component, a stated limitation of RQ1.
6. Claude-local labels (brief-13): those contrasts are labeled scaffold-confounded, naming every
   known difference: Claude runs at the CLI's default temperature, with no reply cap (the
   384-token cap reaches only the local server; Claude run specs record `max_tokens` 0, not
   applied), and with each step sent as one role-tagged prompt. Tier pairs under one harness
   compare equal timeouts on every task (review code-5).
7. Cell labels and warnings (brief-3, brief-12, code-11). Each E0 cell's method and interval type
   name the function and label the library checks, so a failing cell makes the function E1 calls
   warn: `glmm_task_jackknife` for fixed contributions, `glmm_plugin_share` for shares,
   `glmm_mde` for MDE rows (checked in `mde_tables`), `contrast_difference` for harness
   differences, `sampling_discordance` for the sampling table. `glmm_task_jackknife` and
   `glmm_plugin_share` are GLMM intervals and take G5's 500-replicate floor, as D-017 read bca.
   `decompose`, `policy_bounds`, `rank_confidence`, and `cluster_robust_mean` now check validity,
   and G5 checks that each failing cell's own function warns.
8. Shares over jackknife boxes (code-7): the pilot-7 rows were scored with plugin shares mapped
   from the task-jackknife boxes, fixed-factor shares only (pilot 7's two bootstrap replicates
   give the random components no interval). Both pools meet the D-028 rule; the check is in
   `artifacts/method_pilots/summary.json` (exploratory, added after pilot 7 ran).
9. A replaced method (brief-11, resolution-5): an amendment frozen before the first E1 run may
   replace a failing kind's method only beside new headline cells of that kind, which G5 then
   requires to pass; the replaced cells move to a `retired-headline` block, stay in the grid, and
   are listed with a warning if they fail. Otherwise a failing headline kind fails G5.
10. Code pins (ops-1, ops-2, ops-7, resolution-7). Every simulation row names the analysis code
   (a hash of the files that compute E1's intervals and their validation, the lock, and the
   libraries); the G6 and E0 part 2 artifacts record it, and the E1 analysis and the post-E1
   check refuse other analysis code unless a frozen amendment of 0004 allows it. Each E1 batch
   compares its run code (the `src/` tree, the lock, and the llama-server binary) with the arm's
   first banked run: a change stops the arm until a resume accepts it as a deviation, and
   uncommitted changes block every batch. G6 and E0 part 2 refuse to add rows beside rows from
   another environment; a study with rows from two environments runs again in one. (D-034
   items 8 and 9 widen the pinned files and correct what the server digest covers.)
11. Server settings (ops-8): the model pins include each model's server arguments (thinking off
   for both Qwen3 models) and the context per slot, the `e1-design` block pins the llama-server
   build and its binary's SHA-256, and the local arm's first batch needs that binary (corrected
   by D-034 item 9: the SHA-256 covers the binary and the shared libraries beside it).
12. Claude slice (code-9, ops-3, ops-5, ops-9, ops-10, resolution-6). The slice checks its stops
   before every (tier, task) pair; three infra_error runs in a row stop it (those runs stay
   banked); a Claude Code child that did not start clean stops it, and that run is left out and
   counted; the slice starts only after `python -m noisefloor.claude_arm smoke` prints SMOKE:
   PASS, and a new CLI version is accepted only after a smoke run on it passes. A tier served by
   another model ID ends that tier only (revising D-030 item 1): its later runs are left out and
   counted, and the other tiers go on. The slice ends when its runner records an end point (every
   planned run of the tiers still served is banked, or 45 days after the freeze, or an early end
   named as a deviation); the analysis machine's clock does not end it. Its stops, resumes, and
   ends are written only where the billing guard is clean, that is, on the machine that runs it.
   Neither arm starts before the G6 and E0 part 2 artifacts are banked.
13. Local arm (code-2, code-10, code-12, ops-4, ops-6, ops-13, ops-14). The first batch starts
   within 15 minutes after 00:00 UTC. After a batch aborts, every run recorded later is dropped,
   whatever its status. A defect stop takes effect when the running batch ends. A relaunch never
   runs beside a live runner of the same arm (the runner holds a lock, and its model server gets
   SIGTERM if the runner dies). (Corrected by D-034 item 10: that lock covered one planned batch
   only, and the sandboxed runner did not die with a killed orchestrator.) The analysis treats the arm as ended once the clock passes its day
   7. A defect stop resumes only under a frozen amendment of 0004. The batch term of the
   sensitivity fit is the planned batch; day and block are those of each executed piece's start.
14. Analysis (code-1, code-3, code-4, code-15, brief-6, stats-6, stats-8, resolution-8). The
   post-E1 check covers the MDE rows; the 10 percent failed-replicate rule covers the MDE rows,
   the tasks-needed rows, and the residual share; a fit no optimizer completes leaves the
   decomposition and MDE tables "not estimable" and every other table banked; the confirmatory
   flag needs the chain SHA-256; E3 on E1 adds the ranking under each policy and bounds on
   adjacent pairs (unvalidated); the independence reading of the sampling table is frozen (the
   mean over models of the assumed E[p (1 - p)] above the upper end of the reference row's
   interval); the post-E1 check plants an auxiliary fit with its task-by-factor variances, scores
   the task truth the primary model targets, and plants the upper end of the batch variance.
15. Simulation (brief-14, code-13, stats-15): the simulator deals replicate placeholders into
   batches with the same function as `local_plan`, and a test shows that the G6 analysis and the
   E1 analysis code give the same estimates and intervals on one simulated dataset.
16. Validated ranges (brief-10, code-6, stats-5, stats-9, stats-10, resolution-12): the
   `validated-range` block gives each headline kind's span on the scale E1 compares (differences
   in absolute value), and the E1 analysis flags each estimate outside it. Cells extend the spans:
   local groups with model-by-harness effects and with two equal models, Claude tier and harness
   cells at larger gaps, Claude-local cells at low local rates, bound cells at larger ambiguous
   shares, and sampling cells that simulate the sampling table's own data.

## D-034 2026-11-04 Final review of 0004 to 0006 (before freezing)
The final review of the third drafts (three lenses, then adversarial verification) confirmed 40
findings. These changes answer them before the freeze and before G6 and E0 part 2 run, so the
analysis-code pin fixes the corrected code; none rests on E1 data, which do not exist.
1. Sample size (final-stats-5, stats-6). The sizing rule of D-033 item 1 simulated normal test
   statistics, while the local families are tested with `cluster_t` on tasks - 1 degrees of
   freedom. The Holm power rows now simulate each family's own test statistic (t, with its own
   chi-square draw per member for the estimated standard error, for the `cluster_t` families;
   normal for the Wilson families), and the single-comparison power of a `cluster_t` row is the
   noncentral t power. Under t the rule gives 17 tasks per family, 85 tasks; the local arm runs
   3 models x 85 tasks x 18 runs over the same 4 days x 2 blocks (the `e1-design` block's slot
   estimate stays inside the 12-hour blocks). Each power row names its test, and
   `tasks_for_power_0.8` is described as the tasks one comparison needs (the Holm reading needs
   more).
2. Validated spans (final-stats-7): a difference kind's span starts at 0 when one of its cells
   plants no difference (e1-local-mh harness m0, e1-local-eq pairs m0-m1). Those truths come from
   one simulated dataset and lie near 0 on either side, so a positive lower end was seed noise.
3. Batch rule (final-stats-2, res-7). Items 9 and 10 of 0004 now use `batch_var_upper`, the upper
   97.5 percent quantile of the batch-term fit's finite bootstrap replicate batch variances,
   which the replicates bound. The `basic_log` upper end, estimate^2 over the lower quantile,
   grows without bound as that quantile nears 0. A batch-term fit without that value (more than
   10 percent of its replicates failed) counts as failed: the post-E1 check plants 0 and reports
   concurrency, day, and block as not validated.
4. Post-E1 check (final-stats-8, stats-9, res-6, ops-5): the MDE truth uses the task truth of the
   other components (the planted task variance plus half of each task-by-factor variance); the
   simulated batches include the replicate placeholders, as in E1, G6, and E0 part 2; `run` and
   `bank` take `--amendment`; `post_check.json` records its analysis code and amendment; and G7
   requires `post_check.json`.
5. Labels (final-stats-3, res-2, ops-3, res-13). The Claude-local rows are labeled
   `scaffold-confounded` with a `confounds` list. The analysis also writes each Claude row's
   complete pairs and `matches_e0_cells`. The labels that rest on other banked artifacts
   (underpowered families from `family_verdicts`, the fixed factors' power from G6's
   `above_zero`, and a kind resting on an unsupported method) are applied by the report step
   from those artifacts, not by the analysis.
6. Independence reading (final-stats-10): kept as drafted, and 0004 states that it treats the
   implied E[p (1 - p)] as known and has no stated false-positive rate. Comparing the measured
   interval with an interval for the implied value would make a false "violated" rarer at a cost
   in power to see dependence, and the reading changes labels only.
7. Sampling estimate (res-14): 0004 defines it as the code computes it, the mean over tasks of
   the per-task mean of (y1 - y2)^2 / 2.
8. Code pins (res-15, ops-9, ops-8). `ANALYSIS_CODE` adds `evalstats/validity.py`,
   `noisefloor/e0.py` (truths and simulators), `e0_run.py` (cell hits), `e1.py` (plan and
   exclusion rules), and `models.py` (model families); the library's list of failing cells stays
   out, since it is rebuilt after E0 part 2 by design. An amendment of 0004 allows other analysis
   code only through a ```json analysis-code``` block naming that code's hash, and a defect
   stop resumes only under an amendment whose ```json defect-resume``` block names the stop.
9. llama-server (final-stats-1, res-1, ops-1, ops-13): the pin is the digest of the binary and
   the shared libraries beside it (`noisefloor.models.server_build_sha256`), and the freeze
   asserts that it equals this machine's build. The build lives in a cache a new container does
   not keep, so a rebuilt server may start the local arm only under a frozen amendment of 0004
   whose ```json llama-server``` block names its digest, a deviation.
10. Runner (res-8, ops-2). `unshare --fork` blocks SIGINT and SIGTERM, so the parent-death
   SIGTERM added in f4a1046 stayed pending, and a killed orchestrator left its runner and model
   server alive. The sandbox now uses `setpriv --pdeathsig KILL`, tested by killing a real
   launcher. No batch starts while any runner under the work root holds its lock, not only a
   runner of the same planned batch.
11. Claude slice (res-3, ops-6, res-4, ops-7, res-9): the 45-day end point is recorded whatever
   stop is open; three infra_error runs in a row anywhere in banking order since the start or
   last resume stop the slice, also when they end inside a pair; the first banked CLI version
   counts as accepted only if a smoke run on it passed; 0004 states that the clean-start check
   covers the Claude Code harness (minimal-agent steps are single calls without tools or MCP
   servers, and every stream refuses an API-key login).
12. Order and containers (res-5, ops-11, ops-13). G6 and E0 part 2 write all their rows before
   either is banked; `e0_validity` then runs, and its commit, the one change under `src/` from
   the start of G6 to the E1 analysis, lands before the local arm's first batch and before the
   Claude clone. Rows are committed and pushed at check-ins. Runs banked after the last push and
   lost with a container are noted in the stop log (`python -m noisefloor.e1 note`, which stops
   nothing) and run again under their planned run ids, a deviation.
13. Gates (final-stats-4, ops-4, res-11, ops-12). G5 accepts a retired headline kind only beside
   new cells that test another method (for a GLMM kind, another rule in the amendment's own
   recovery-config) and a frozen amendment of 0004 from before E1 data. G7's recompute runs four
   GLMM fits with bootstrap and jackknife and gets a 6-hour timeout. G6 requires a 16-hex
   `analysis_code` equal to E0 part 2's once that is banked. G8 requires the rankings and
   adjacent-pair bounds of E3 on E1.
14. Texts (ops-10, res-12, ops-14, res-10, ops-15): 0004 and 0005 state that replacing a kind's
   method takes amendments of 0003, 0004, and (for a GLMM kind) 0006; 0006 names 0004's
   fixed-factor reading; the E0 coverage artifacts are labeled synthetic; CORRECTIONS.md records
   the withdrawn and corrected claims.
15. Grader (found by a property test during `make prepush`, not by the review). `grade_answer`
   scored an answer that matches as submitted as wrong when extraction changes it, which
   happens when the expected answer itself looks like a wrapper (`_0_` reads as markdown
   emphasis). The score is now 1 when the answer matches as submitted or after extraction
   (D-021's rule otherwise). Every expected answer of the suite is a fixed point of extraction
   (checked on all four answer families, three difficulties, and seeds 100000 to 100999 and
   900000 to 900999; a test pins the E1 seed range), so for every suite task a match as
   submitted already implied a match after extraction, and no grade of a suite task changes.
16. Verification of these fixes (one verifier per finding group and a regression reader; the
   stats and resolution groups finished and found 24 of 25 findings fixed). Follow-ups: G5's
   retirement rule counts only an amendment of 0004 with an analysis-code block (a llama-server
   amendment no longer satisfies it), and a fixed contribution's rule is the task jackknife or
   the bootstrap box with its `box_interval`, so changing `box_interval` under the jackknife is
   no new method. G6 compares its analysis code with E0 part 2's only for the base 0006
   artifact; a rerun under an amendment of 0006 (a replaced GLMM method) must run code that a
   frozen amendment of 0004 allows. The arms refuse to start unless the analysis code at HEAD,
   recomputed with each artifact's own library versions so the check also holds on Raj's
   machine, is the code G6 and E0 part 2 recorded (or an amendment allows it), and unless the
   library's list of failing cells is what `e0_validity` builds from the banked coverage.
   Pinning `e1.py` and `models.py` keeps them unchanged through G7's recompute; 0004 and PLAN
   now say that a repair to them during an arm needs an analysis-code amendment.

## D-035 2026-11-11 Public release: README, an exploratory noise floor, and a G13 fix
1. README.md replaces the build brief (kept as docs/BUILD_BRIEF.md), as section 11 asks once
   real results exist. Its results are the preregistered E2 and E3 reanalyses and E0 part 1;
   E1 is described as preregistered and not yet run. Every number in it is a registered claim
   or an exact-text exemption, and a fresh-eyes review (a lab reader and a statistician) was
   applied before release.
2. The noise floor of each board's own design is exploratory, not preregistered:
   `tools/noise_floor.py` derives, from the banked E2 paired intervals only, the true gap each
   adjacent pair's design detects with power 0.8 at two-sided alpha 0.05 (normal approximation)
   and banks per-board medians in `artifacts/e2/noise_floor.json`, which `make reproduce`
   checks. It is labeled exploratory wherever it appears.
3. `make report` adds a figure per board of every adjacent gap with its paired interval (the
   quantity of prereg 0001's first question); the Terminal-Bench one heads the README.
4. G13 checked the first-screen interval on raw Markdown, where the claim markers G10 requires
   sit inside the brackets, so no claim-marked README could pass both gates. It now removes the
   markers, which do not render, before matching; the property it checks is unchanged.
5. `make reproduce` is implemented: it recomputes E2 and E3 from the pinned snapshots,
   regenerates the figures and the noise-floor file, and fails if any committed one changes.
6. The policy-bounds interval of a success rate is cut to [0, 1] (a reviewer saw an upper end of
   1.001 in the Terminal-Bench report); E2 and E3 are re-banked with that one change, recorded in
   CORRECTIONS.md.

## D-036 2026-11-11 Operating rules for the validation runs and E1, from a pre-mortem (before acting on them)
A read-only pre-mortem (five lenses, each finding checked by an adversarial verifier) of the G6
and E0 part 2 runs, their banking, the E1 local arm, resources, and the Claude runbook found
procedure gaps and no defect that changes a result. These rules hold from now on. None changes
`src/`, a preregistration, or a gate; the checks live in `tools/runcheck.py`,
`tools/e0g6_driver.sh`, `tools/wait_e0g6.sh`, and `tools/wait_e1.sh`.
1. Launch condition. Neither arm reads coverage verdicts before it starts (`require_validation`
   checks that the artifacts exist, the analysis code, and the list of failing cells), and the
   first confirmatory run closes the preregistered route for replacing a failing method (0004,
   0005). So the local arm launches only when the fast ship-check at the banking commit prints
   G5 PASS and G6 PASS, with `src/evalstats/data/validity.json` and
   `artifacts/headline_cells.json` committed. Otherwise no arm starts, Raj is asked not to start
   the Claude slice, the hold is recorded, and amendments of 0003, 0004 (an analysis-code block),
   and for a GLMM kind 0006 are frozen before any confirmatory run.
2. The E1 analysis waits for both arms (prereg 0004: it "refuses to run until both arms have
   ended"). The local arm alone yields no E1 result: the build, E3 on E1 (G8), the post-E1 check
   (G7), the report's E1 sections, and the final review (G12) start once Raj's end point for the
   Claude slice is in this repository. When the local arm has ended and nothing else remains,
   STATUS reads BLOCKED ON RAJ. `python -m noisefloor.e1 claude` is never run in this hosted
   session: it can write the slice's stop log before the billing guard refuses a call.
3. CPU and memory during E1, an interpretation of brief 10.2 steps 1 and 4. E1 runs have
   wall-clock limits, every shell command gets a scheduler autogroup with the same weight as the
   model server (D-024), and a Qwen3-4B server at concurrency 2 leaves about 4 GB of memory.
   While the local arm runs a batch, a loop iteration or heartbeat is the check on a background
   run: it reads the process, the log, `free -m`, and `df`, and commits banked rows. The fast
   ship-check, tests, lint, typecheck, `make ci` or `make prepush`, R, report builds,
   simulations, and workflows run only after `python3 tools/runcheck.py e1-gap --need <s>`
   prints `E1-GAP: OPEN` (the arm waits between blocks and the job ends before the next 00:00
   or 12:00 UTC boundary less 10 minutes), or after the arm ends. During the arm every shell
   command starts with `echo 19 > /proc/self/autogroup; echo 1000 > /proc/self/oom_score_adj`,
   so it yields the CPU and is the process killed if memory runs out. The arm itself is launched
   from a fresh shell with `setsid`, so its autogroup keeps nice 0.
4. Launch window and keep-alive. The arm is launched between 21:00 and 23:45 UTC, after the
   RELEASE-READY update and every commit. The launch checks run once, so nothing in git changes
   until the first batch is banked; a change in that window means kill and relaunch. The
   waiter `tools/wait_e1.sh` runs as a harness-tracked task, restarted first on every wake-up,
   and its two-hour limit must not fall between 23:45 and 00:30 UTC on launch night; an hourly
   scheduled check-in fires into this session while the arm runs; after a restart or stop the
   relaunch comes before anything else. If the arm dies in its first batch and cannot be
   relaunched before 00:15 UTC, the runner waits for the next 00:00 UTC and the first batch's
   completed runs become a day of their own; that is recorded with `python -m noisefloor.e1 note
   local` and reported as the off-plan schedule 0004 describes.
5. Relaunching the local arm: `python3 tools/runcheck.py e1-relaunch` refuses while an arm,
   batch runner, or model server is alive and lists each unbanked workspace; `--rotate` moves
   aside the output of a workspace from which nothing would be salvaged, so the retry does not
   replay the cached model replies of the killed attempt (0004: interrupted runs run again from
   the beginning). An `aborted.json` it lists is recorded with `e1 note local`. Then the same
   launch command.
6. Relaunching G6 and E0 part 2: only through `tools/e0g6_driver.sh`, which runs `python3
   tools/runcheck.py e0g6-preflight` first (no other driver or study process; no uncommitted
   change under `src/`, `uv.lock`, or `pyproject.toml`; a cut final line in a raw row file is
   removed and printed, which 0005 and 0006 allow, and noted in docs/LOG.md; every line is a JSON
   object; the environment is the one the rows recorded) and stops at the first step that
   fails. The running `runs/e0g6/driver.sh` (git-ignored) goes on to the end. The waiter
   `tools/wait_e0g6.sh` also exits when no raw row file has grown for 45 minutes.
7. Nothing under `src/` changes in the main checkout from now until both arms have ended, apart
   from the `e0_validity` commit: rows from a dirty tree are refused only at banking, and an
   uncommitted file under `src/` makes the E1 arm exit at its next batch boundary. Fixes are
   drafted in a separate worktree outside the checkout.
8. Before banking, completeness is read from the raw rows, not the log: every G6 seed, every
   replicate index of every 0005 cell, and no step exit after the last start line. The bank
   commands are exactly those in STATUS.md (`e0_run bank` with 0003 first and `--only`). The
   generator strings that `recovery.json` and `coverage-0005.json` record are not runnable as
   written; fixing them would change pinned files, so they are documented here. `e0_validity`
   also rewrites `artifacts/headline_cells.json`, so the REPORT claim that counts headline cells
   is updated in the same commit, with a CORRECTIONS entry if its number changes.
9. Review flow (G12). The brief sends non-blocking issues to docs/REVIEW_LOG.md and lets a
   reviewed commit differ from HEAD only in `artifacts/review/`, STATUS.md, and docs/LOG.md, so a
   reviewer that appends to REVIEW_LOG.md makes its own review stale. The reviewer now writes
   only `artifacts/review/latest.json` and reuses the id of an issue the log already lists; the
   main session copies new non-blocking issues into the log in the first commit of the next
   final-sequence turn and reviews again. The brief says where the issues go, not who writes them.
10. Deferred changes under `src/`: rotating the output of any unbanked workspace in
   `orchestrate._prepare`; treating a salvaged workspace that has an `aborted.json` as a failed
   batch; a client latch after a paid-extra-usage refusal; a guard in `report.e1_mde` for a fit
   that is not estimable; refusing to start the validation runs from a dirty tree. Each waits
   until both arms have ended (a commit under `src/` during an arm records a code change) unless
   a later entry says otherwise. Changes in pinned files (counting a started first batch as the
   arm's start, re-raising a missing R in `recover`, comparing E0's environment with G6's) would
   need a frozen amendment of 0004 with an analysis-code block and are not planned.

## D-037 2026-11-11 Follow-ups to D-036: blinding in the tools, the G14 turn check, the stack
1. Blinding. Prereg 0004 lets only the runners read `artifacts/e1/local.jsonl` and
   `claude.jsonl` before an arm ends. So `tools/runcheck.py e1-relaunch` counts the whole lines
   of a workspace's results file and never parses a record, and it reads only the size of
   `local.jsonl`. For the same reason G5's test of "before E1 data" keeps reading only the
   analysis output (`artifacts/e1/runs.jsonl`): it bites once the analysis is built, the final
   ship-check comes after that, and amendments are frozen before any arm starts (D-036 item 1).
2. G14 turn check. `tools/budget.py` read turns from `agent_run` rows, which the client writes
   without turns, so it always found 0. It now reads the `agent_usage` rows the client writes. A
   Claude Code child runs with `--max-turns` at the cap, and the stream check stops it at the
   first message of a turn over the cap, which records cap + 1 turns: that is reported as
   stopped at the cap, and only more fails. Other rows that carry turns stay held to the cap.
3. `make doctor` has a `model routing` row naming `ANTHROPIC_*` variables outside the guard's
   list and settings `env` blocks that set `ANTHROPIC_*` or `CLAUDE_CODE_USE_*` names (names
   only, never values). They could remap the model aliases on Raj's machine, and the billing
   guard that the client enforces does not cover them; the row reports and blocks nothing.
4. The analysis-code pin includes the library versions, so the E1 analysis must run on this
   stack weeks from now: Debian packages r-base-core 4.3.3-2build2, r-cran-lme4 1.1-35.1-4,
   r-cran-matrix 1.6-5-1, libblas3 and liblapack3 3.12.0-3build1.1 (the reference BLAS at
   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3), python3.12 3.12.3-1ubuntu0.17, and the locked
   `.venv`. Nothing here runs `apt upgrade` or rebuilds `.venv` before the analysis. A new
   container needs these exact versions (Needs Raj item 3); otherwise a frozen amendment of 0004
   with an analysis-code block comes before the build.
5. The post-E1 check reuses the planted configuration it wrote first, so the E1 tables are
   built once, after both arms have ended and Raj's files are merged; before any rebuild,
   `artifacts/e1/post_check/` and `artifacts/e1/post_check.json` are removed.
6. G13 caps the REPORT at 8,000 words, and about 1,400 are left. The E0 part 2, G6, and E1
   results replace the report's "not run" passages instead of being added beside them.
7. `tools/validity_summary.py` turns the banked E0 part 2 and G6 artifacts into per-cell and
   per-component lists of verdicts, using the coverage rule and hit test of
   `tools/ship_check.py` itself, so the REPORT can count passing cells and components from an
   artifact (claims count lists, not maps). Its outputs are labeled synthetic, it skips a source
   that is not banked yet, and `make reproduce` checks them.
8. The Claude slice starts only once STATUS.md records G5 PASS and G6 PASS at the banking
   commit, for the reason in D-036 item 1 (the slice's own start check does not read coverage
   either); the runbook and Needs Raj item 1 say so.
9. Before the final sequence a milestone review runs and its non-blocking issues are copied into
   docs/REVIEW_LOG.md, so the final review is likely to raise none. With D-036 item 9 a review
   that raises a new non-blocking issue reads `REVIEW: STALE`; at final-sequence step 2 that
   ends the turn as a BLOCK does, and the sequence starts again in a later turn.

## D-038 2026-11-12 Last findings of the D-036 pre-mortem
The verifiers checked the remaining findings against the branch after D-036 and D-037; most were
already fixed. Four small ones are applied here (docs and `tools/` only):
1. The E1 slot estimate is low. The dry run behind prereg 0004's `duration_estimate` ran no
   violation-family task, the slowest family in screening, so slots may run about 10.5 to 12
   hours and leave gaps between blocks of about an hour or none, not 2.2 to 2.5 hours. Nothing in
   the design changes: a batch that cannot start in its block moves to the same block of the
   next day and is reported as off its planned slot, as 0004 says. Pushes are planned for short
   gaps, and after day 1 the slots are re-forecast from the timestamps in `runs/e1/alive.txt`.
2. `tools/runcheck.py e1-gap` read the first-batch wait line as a gap up to the moment the
   runner process starts. It now reports BUSY from 00:00 to 00:15 UTC while that line is the
   last one.
3. A dirty tree during an arm. The runner logs "commit first" and exits without a stop row, and
   committing the change records a code change. `git restore` of the paths alone leaves staged
   edits and untracked files, which the runner's check also counts. So the remedy in PLAN for
   both arms is `git restore --staged --worktree -- src uv.lock pyproject.toml`, then `git clean
   -n -- src` and removing what it lists, never a commit. `tools/runcheck.py e1-relaunch`
   refuses a dirty tree, and `.gitignore` now ignores `.DS_Store` and editor swap files, which
   would otherwise count as untracked files under `src/` on Raj's machine.
4. The client latch after a paid-extra-usage refusal (D-036 item 10) would help only before the
   slice starts, and landing it then would add a second change under `src/` to the window that
   D-034 item 12 keeps for `e0_validity`. It is dropped. Rule 5 rests on the runbook instead:
   extra usage off for the whole slice, and no further call after a refusal that names it.

## D-039 2026-11-12 Gate checks tightened after the milestone review
The milestone review (D-037 item 9; REVIEW_LOG 2026-11-12) found one blocking issue, a literal em
dash in a test (rule 12), and fourteen non-blocking ones. Changes in `tools/` and `tests/` only,
each one stricter than before (docs/GATES.md):
1. A test scans every tracked file we write (all but `artifacts/` and recorded third-party
   fixtures) for em dashes.
2. G12 reads the verdict from stdout alone (uv can warn on stderr after it) and matches review
   ids as whole ids, in G12 and `tools/review.py`.
3. G6 requires each banked dataset to equal its committed raw row; the raw rows to be exactly
   the preregistered seeds, from the artifact's environment and analysis code and a clean
   `src/`; and the planted values that the recovery-config fixes (task, model-by-task, and model
   variances; each two-level factor's contribution, effect squared over four) to be the ones
   every dataset was scored on. The other components' planted values depend on each dataset's
   own draws and are not restated in the gate.
4. G5 requires each counted raw line of a grouped cell to carry `e0_glmm_seed(group, i)`
   (restated in the gate; a test pins it to the study's function) and, for an artifact banked
   with stamps, every counted line to name that environment and analysis code from a clean
   `src/`.
5. G14's turn check is strict: any `agent_usage` row above the cap fails it, including a run the
   stream check stopped at the first message of a turn over the cap. This replaces D-037 item 2,
   which reported that case as stopped at the cap: a turn that began is past the cap.
6. G2 rejects every inline `mypy:` setting comment; G1 fails when a test was skipped for want of
   R, lme4, or rpy2; G10 also runs the committed rebuild check of each derived artifact that
   claims cite (the noise floor, the method-pilot summary, and the validation summaries).
7. Deferred: N2, marginal rank intervals (`evalstats.rank`, `simultaneous=False`) use the other
   entry's critical value on the best-rank side. No reported number rests on them, and the file
   is in the pinned analysis code, so the fix lands after the E1 analysis is banked, with E2
   banked again and a CORRECTIONS entry for every banked interval it changes.

## D-040 2026-11-12 Further checks from the milestone review's completeness critic
1. `artifacts/e3/bounds.json` backs 20 REPORT claims but was compared with nothing that
   recomputes it. Each of its public-board entries equals that board's `results.json` q3 for its
   analysis set, plus `source` and `analysis_set`, so G8 and a test (`tests/test_banked.py`) now
   require that equality: G8's full mode recomputes `results.json`, and the tie carries that to
   every E3 number.
2. The leaderboard reports are rendered from their banked files in a test, and `make reproduce`
   regenerates them before its diff, which could not fire before.
3. The cross-check tests compare their fresh results with `artifacts/e0/crosscheck.json`, with
   names, tolerances, and reference implementations exact and values to 1e-6: lme4's fit moves by
   about 1e-7 from run to run, the same on both of its paths, so each check's own difference
   stays near 1e-12.
4. G14 passes the sensitive variables that `clean_env` removes to `make secrets-scan` alone, so it
   can search for their literal values; the scanner only reads files and prints pattern names.
5. Recorded, not changed: G5 counts the simultaneous rank cells' stored verdicts, because their
   raw rows hold no ranks; and the E2, E3, and E0 part 1 artifacts carry no git SHA, code hash,
   environment, or command line (brief 10.2 step 5). Both are addressed when E2 is banked again
   after the E1 analysis (D-039 item 7), with per-model ranks and provenance fields in the files.
6. Section 11 asks for a 60-second quickstart that runs offline. It is read as: after a one-time
   clone and `make setup`, which need the network, the quickstart runs with no network (the
   README says so), since nothing can be installed without it.
7. Rule 8 was checked: PR #1 was opened and merged from Raj's account, and this session called
   no pull-request tool.
