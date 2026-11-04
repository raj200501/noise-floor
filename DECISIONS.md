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
