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
- N7: G5 now requires each counted raw line of a grouped cell to carry its dataset's seed and,
  for an artifact banked with stamps, each line to name that environment and analysis code from
  a clean src/. Fixed (D-039).
- N8: review ids are matched as whole ids in G12 and `make review`. Fixed.
- N9: the G14 turn check is strict again: a run that began a turn past the cap fails it, even
  when the stream check stopped it at once. Fixed (D-039 replaces D-037 item 2).
- N10: G2 rejects any inline mypy setting comment, not only `ignore-errors`. Fixed.
- N11: G10 now runs the committed scripts that rebuild the derived artifacts claims cite
  (`tools/noise_floor.py --check`, `python -m noisefloor.method_pilots --check`,
  `tools/validity_summary.py --check`). Fixed. The E0 cross-check artifact has no rebuild mode;
  its four claims stay covered by freshness.
- N12: docs/DATA.md lists the model weights: repository, revision, file, license read from each
  model card at the pinned revision, retrieval date, and where each was used. Fixed.
- N13: STATUS usage and claim counts updated. Fixed.
- N14: G1 fails when any test is skipped for want of R, lme4, or rpy2, since the preregistered
  GLMM checks are among those tests. Fixed (D-039).
