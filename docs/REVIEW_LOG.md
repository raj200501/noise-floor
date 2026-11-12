# Review log

Non-blocking issues from each `make review`, by id and date. Blocking issues are fixed before
shipping and recorded in artifacts/review/latest.json.

## 2026-11-12: milestone review of efbff39 (DECISIONS.md D-037 item 9; verdict BLOCK)

Blocking: B1, a literal em dash in tests/test_e2.py:168, fixed, with a test that scans every
authored tracked file (D-039). The non-blocking issues, with what was done:

- N1: G12 compared the last line of stdout and stderr together, so a uv warning made it fail
  every time here. Fixed: it reads stdout (D-039).
- N2: marginal rank intervals (`simultaneous=False`) test an entry against another entry's
  critical value on the best-rank side. They back no reported number. `evalstats/rank.py` is
  in the pinned analysis code, so the fix waits until the E1 analysis is banked, then E2 is
  banked again with a CORRECTIONS entry (D-039).
- N3: the REPORT and README said G6 and E0 part 2 had not run while G6 was running. Fixed.
- N4: E3's robust counts now say, next to the counts, that their interval has no coverage cell
  of its own (README and REPORT abstract). Fixed.
- N5: the [0, 1] cut of the policy-bounds interval is now listed under the REPORT's deviations
  from preregistration. Fixed.
- N6: G6 now requires each banked dataset to equal its committed raw row, the raw rows to be
  exactly the preregistered seeds from one environment, analysis code, and a clean src/, and
  the planted values the recovery-config fixes to be the ones every dataset used. Fixed (D-039).
- N7: G5 now requires every raw line of a counted index to carry its dataset's seed (grouped
  cells) and, for an artifact banked with stamps, to name that environment and analysis code
  from a clean src/, as the bank requires; an amendment's artifact must name both. Fixed (D-039,
  D-040).
- N8: review ids are matched as whole ids in G12 and `make review`. Fixed.
- N9: the G14 turn check is strict again: a run that began a turn past the cap fails it, even
  when the stream check stopped it at once. Fixed (D-039 replaces D-037 item 2).
- N10: G2 rejects any inline mypy setting comment, not only `ignore-errors`. Fixed.
- N11: overstated in part. G1's tests already rebuild the noise floor, the method-pilot
  summary, and the figures and compare them with the banked files; the derived files that no
  check compared were `artifacts/e3/bounds.json`, `artifacts/e0/crosscheck.json`, and the
  leaderboard reports (the completeness critic, below). G10 also runs the rebuild checks named in
  docs/GATES.md, a harmless overlap. Fixed (D-039, D-040).
- N12: docs/DATA.md lists the model weights: repository, revision, file, license read from each
  model card at the pinned revision, retrieval date, and where each was used. Fixed.
- N13: STATUS usage and claim counts updated. Fixed.
- N14: G1 fails when any test is skipped for want of R, lme4, or rpy2, since the preregistered
  GLMM checks are among those tests. Fixed (D-039).

Found by the review workflow's completeness critic, beyond the reviewer's list:

- B2 (blocking, fixed): nothing compared `artifacts/e3/bounds.json`, which backs 20 REPORT
  claims, with a recomputation; on a scratch copy an edited E3 count passed claimcheck,
  freshness, and G8. G8 and a test now require each entry to equal its board's results.json q3,
  which G8's full mode recomputes (D-040). No reported number was wrong.
- N15: the leaderboard reports were never regenerated or compared, and `make reproduce`'s diff
  over docs/leaderboards could not fire. A test now renders each from its banked files, and
  `make reproduce` regenerates them before its diff. Fixed.
- N16: under ship-check the secrets scan ran without the sensitive variables, so it could not
  search for their literal values. G14 now passes them to the scan, which prints names only.
  Fixed.
- N17: G5 cannot recount the three simultaneous rank cells, whose raw rows hold the simulator's
  own verdicts. Recorded as a limitation in docs/GATES.md; storing per-model ranks waits until
  after E1 (D-040).
- N18: the E2, E3, and E0 part 1 artifacts carry no git SHA, code hash, environment, or command
  line (brief 10.2 step 5). They reproduce through the recomputes; provenance fields are added
  when E2 and E3 are banked again for N2 (D-040).
- N19: `artifacts/e0/crosscheck.json` was recomputed by the tests but never compared with the
  banked file. The cross-check tests now compare (values to 1e-6, the run-to-run drift of
  lme4's fit being about 1e-7). Fixed.
- Rule 8 check: PR #1 was opened and merged by Raj's account (GitHub, 2026-11-11T20:16:56Z and
  20:17:52Z); this session called no pull-request tool.

