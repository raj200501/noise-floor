goal_turn: 0
STATE: ACTIVE

# Status

Updated 2026-10-23, bootstrap (heartbeat turn; the `/goal` loop has not started, so goal_turn is 0).

## Gates

Filled from the fast ship-check after the bootstrap commit.

## Now

Bootstrap (docs/BUILD_BRIEF.md section 10.1).

## Next three actions

1. M1: evalstats long-format schema, generic CSV/JSONL reader, paired bootstrap, cluster-robust
   SEs, analytic power and MDE, CLI `compare` and `power`, tests to 85% (G1, G2, G3).
2. M2: offline demo on labeled synthetic fixtures with golden output (G4, tag v0.1).
3. M3: prereg chain, claimcheck, freshness (port from raise-v1 with attribution) (G10, G11).

## Environment (make doctor, 2026-10-23)

- Linux x86_64 container (Claude Code on the web), 4 CPUs, 15.7 GiB RAM, about 30 GiB free disk
  (floor 8 GiB), no GPU. Python 3.13 system; 3.11 and 3.12 available; uv 0.11.
- Docker binary present, daemon not running: Docker suites unusable, Docker-free fallback.
- No R or lme4: GLMMs need a named Python backend (or an R install, to try).
- No Ollama or llama.cpp: local model backend unusable until a runtime and models are installed.
- OS sandbox: `unshare -rn` works (network namespace); write confinement still to build (D-006).
- `claude` CLI 2.1.309 present, but the billing guard is tripped by `ANTHROPIC_BASE_URL` and
  `CLAUDE_CODE_USE_CCR_V2` (set by the hosting environment): Claude backend unusable here (D-004).
- Free tiers: not confirmed (`FREE_TIER_KEYS_CONFIRMED` unset), ignored.
- Backends usable now: none for model runs. Fallbacks: subagent reviews, local models after install.

## Usage to date

cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0, local model runs 0.

## Needs Raj

1. Claude subject arm (G7 needs at least 150 Claude-tier runs): the billing guard blocks Claude
   children in this hosted session. Action: when E1 is preregistered, run the Claude arm on a
   machine where `make doctor` shows `billing guard clean` and the `claude` CLI is logged in to
   the Max plan (for example your laptop), following the run instructions that will be in
   docs/PLAN.md. Not blocking yet.
2. FYI, pushing: `ALLOW_PUSH` is unset, but this web session designates branch
   `claude/gifted-darwin-k7q99b` and instructs pushing to it, and its container is ephemeral.
   I push commits (never tags, never force) to that branch only (D-003). Say so if you want
   pushes to stop.

## Blocked

None.

## Backlog (suggestions for Raj, not ladder rungs)

1. A `evalstats compare --from-inspect <log dir>` one-liner so an evals engineer gets a paired
   interval from existing Inspect logs without writing any code.
