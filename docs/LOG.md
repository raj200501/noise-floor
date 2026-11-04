# Log

History moved out of STATUS.md, newest first.

## 2026-10-23 to 2026-11-04 goal_turns 1 and 2

goal_turn 1 (2026-10-23): evalstats core (schema, cluster-t and bootstrap intervals, paired
comparisons, power and MDE, rank-confidence sets, policy bounds), logit GLMM decomposition with
lme4 through rpy2 and a statsmodels fallback, offline demo (tag v0.1 local), verified references,
integrity tools ported from raise-v1-public, adapters for Inspect, Harbor, and SWE-bench, and a
llama.cpp CPU build.

goal_turn 2 (2026-11-04): E2 reanalysis of Terminal-Bench 2.0 and Epoch AI SWE-bench Verified
with confidence reports (prereg 0001, amendment 0002); E0 part 1 (prereg 0003, 22 cells) with
library warnings for unsupported cells; cross-checks against statsmodels, scipy, and R; nf-tools
suite, minimal-agent and Inspect harnesses, orchestrator, namespace sandbox; pilot p1 (stopped,
D-018) and screening s2 (stopped, D-020); five examples and the generated API reference; CI
limited to main and pull requests (D-015); commits authored as Raj (D-014). Raj paused the work
to save usage, then asked to continue.

## 2026-10-23 bootstrap

Bootstrap per docs/BUILD_BRIEF.md section 10.1, started by the heartbeat loop in a Claude Code on
the web session. Brief copied and hashed; doctor, scaffold, make targets, ship-check, review
tooling, subagents, plan, and state files created.
