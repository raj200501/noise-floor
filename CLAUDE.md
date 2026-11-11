# noise-floor: working notes for Claude

After any compaction or restart, re-read CLAUDE.md, STATUS.md, and sections 6 to 10 of docs/BUILD_BRIEF.md before acting.

## Mission

noise-floor measures how much of a leaderboard gap is the model. It ships evalstats, a typed
Python library that makes the right analysis of eval results the default (paired and clustered
comparisons, bootstrap and mixed-model intervals, power and minimum detectable effects,
rank-confidence sets, variance components, policy-sensitivity bounds); a preregistered
factorial study that decomposes variance in agentic task success across model tiers and
families, harnesses, resources, time, and sampling; a reanalysis of public leaderboards with
per-task results; and leaderboard confidence reports. The full brief is docs/BUILD_BRIEF.md
(immutable; its SHA-256 is in docs/BUILD_BRIEF.sha256).

## Non-negotiables (docs/BUILD_BRIEF.md section 9, verbatim)

1. **No fabricated or hand-typed numbers.** Every result number (a measured rate, count of outcomes, estimate, interval, or test statistic) in README.md, docs/REPORT.md, other results documents, figures, and tables is produced by a committed script from a banked artifact under `artifacts/`. Design parameters, thresholds, caps, dates, versions, and the text of docs/BUILD_BRIEF.md are not result numbers. `make claimcheck` enforces this and `make freshness` re-derives every live number.
2. **Preregister before confirmatory runs.** Hypotheses, metrics, analysis plan, thresholds, sample sizes, exclusions, and stopping rules go in `prereg/NNNN-*.md`, committed and hashed before any confirmatory data exists. A needed change is a new, dated amendment file that cites the original, is committed before any data it affects, and is reported as a deviation; the original is never edited. Anything not preregistered is labeled exploratory.
3. **Publish failures beside successes.** Null and negative results get the same prominence as positive ones. Every changed or withdrawn claim gets a dated entry in CORRECTIONS.md.
4. **Label synthetic and mock data everywhere.** Synthetic, procedurally generated, or mock results are never presented as model results or real-world results.
5. **$0 cash, with caps enforced in code** (section 8).
6. **Secrets stay in the environment.** Never write keys or tokens to files, logs, artifacts, or command lines. `make secrets-scan` runs in CI and in ship-check.
7. **Defensive scope only.** Mock tools and disposable sandboxes. No real credentials, real systems, or harmful payloads. Build nothing that helps an agent evade graders, monitors, or safety measures beyond what the measurement strictly requires.
8. **No outbound actions.** Never send email, open issues or pull requests, post, publish packages, datasets, or sites, or contact anyone. Pushing is allowed only under section 10.1 step 11. Prepare drafts in `outbound/` for Raj to review and send.
9. **Respect licenses and terms.** Record the source, license, and retrieval date of all third-party data in docs/DATA.md. Do not redistribute what a license does not allow. Add a canary string to any dataset meant for release.
10. **The brief is immutable.** Never edit docs/BUILD_BRIEF.md. Record deviations, substitutions, and interpretations in DECISIONS.md, with a reason, before acting on them.
11. **Verified citations only.** Never cite from memory. Open and check every reference (title, authors, year, URL) once and bank what you saw in `artifacts/citecheck.json` (retrieval date, final URL, page title, content hash); `make citecheck` validates that file offline, so ship-check never needs the network. If a site blocks scripted access, use the arXiv, DOI, or another official copy, and never work around bot protection. If nothing opens, drop the reference and list it under "Needs Raj" for a manual check.
12. **Plain writing.** Precise, scoped claims. No hype words such as "revolutionary", "groundbreaking", or "state of the art". No em dashes anywhere (Raj's style rule).
13. **Honest progress.** Never pass a gate by weakening its check. If a gate cannot pass, say so in STATUS.md and keep working on what can.
14. **Pin and snapshot.** Every public dataset used in a reanalysis is pinned by URL, retrieval date, and hash, and rerunning the reanalysis on the snapshot reproduces every number exactly.
15. **Neutral reporting.** State which orderings are and are not statistically supported. Never speculate about why a submitter's score is what it is.

## State files

- `STATUS.md`: line 1 `goal_turn: N`, line 2 `STATE: ACTIVE | BLOCKED ON RAJ | LADDER COMPLETE`;
  gate table, Now, next three actions, usage, Needs Raj, Blocked, backlog. Under 200 lines.
- `DECISIONS.md`: assumptions, substitutions, interpretations (written before acting on them).
- `CORRECTIONS.md`: dated entry for every changed or withdrawn claim.
- `docs/LOG.md`: history moved out of STATUS.md.
- `docs/PLAN.md`: milestones mapped to gates, risks, fallbacks, usage estimates.
- `docs/GATES.md`: artifact contracts that tools/ship_check.py reads.
- `budget.toml`: usage caps (lower only, never raise). Ledger: `artifacts/usage/ledger.jsonl`.
- `prereg/NNNN-*.md`: frozen preregistrations. `artifacts/`: banked results with provenance.
- `outbound/`: drafts for Raj to send. `private/` and `runs/`: git-ignored.

## Commands

- `make setup` (uv sync --locked), `make doctor` (environment and usable backends)
- `make test`, `make lint`, `make typecheck`, `make ci` (lint, typecheck, test, demo)
- `make demo` (offline, recorded fixtures, golden output), `make report`, `make reproduce`
- `make prereg-verify`, `make claimcheck`, `make freshness`, `make citecheck`
- `make secrets-scan`, `make budget`, `make review` (reviewer subagent fallback here)
- `python tools/ship_check.py --fast` each iteration; the full `make ship-check` only in the
  final sequence (section 10.6). Only a full run counts.
- While the E1 local arm runs (D-036): CPU-heavy work (the fast ship-check, tests, lint,
  typecheck, `make ci` or `prepush`, R, report builds, workflows) waits until `python3
  tools/runcheck.py e1-gap --need <seconds>` prints `E1-GAP: OPEN`; otherwise a check-in only
  reads and commits rows. Shell commands start with `echo 19 > /proc/self/autogroup; echo 1000 >
  /proc/self/oom_score_adj`. After a restart, relaunch first (`tools/runcheck.py e1-relaunch`,
  `tools/e0g6_driver.sh`) and restart the waiter (`tools/wait_e1.sh` or `tools/wait_e0g6.sh`).

## Loop Protocol (section 10.2, short form)

Orient (STATUS.md, fast ship-check unless an E1 batch runs), choose one thing (critical-path
failing gate first), build test-first, verify, bank with provenance, review milestones, record,
commit, raise the bar.
After three failed attempts at one problem, record it under Blocked and switch.
