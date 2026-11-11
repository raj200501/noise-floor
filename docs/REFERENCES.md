# References

Verified bibliography for noise-floor (docs/BUILD_BRIEF.md section 2, rule 11 of section 9).
Each source was opened once with curl on 2026-10-23. Titles, authors, and years are as the page
shows them; for arXiv papers they come from the `citation_title`, `citation_author`, and
`citation_date` tags of the abstract page. The banked record, with the final URL, HTTP status,
and sha256 of each page body, is `artifacts/citecheck.json`, and `make citecheck` validates it
offline. Long author lists are shortened here; the full lists are in that file.

## Verified

1. Gian Segato (Anthropic). "Quantifying infrastructure noise in agentic coding evals."
   Anthropic, 2026 (published 2026-02-05).
   https://www.anthropic.com/engineering/infrastructure-noise
   (redirects to https://www.anthropic.com/research/infrastructure-noise). Retrieved 2026-10-23.

2. Evan Miller. "Adding Error Bars to Evals: A Statistical Approach to Language Model
   Evaluations." arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640. Retrieved 2026-10-23.

3. David Heineman, Valentin Hofmann, Ian Magnusson, Yuling Gu, Noah A. Smith, Hannaneh
   Hajishirzi, Kyle Lo, and Jesse Dodge. "Signal and Noise: A Framework for Reducing Uncertainty
   in Language Model Evaluation." arXiv:2508.13144, 2025. https://arxiv.org/abs/2508.13144.
   Retrieved 2026-10-23.

4. Thomas Kwa. "Clarifying limitations of time horizon." METR research note, 2026 (dated
   2026-01-22). https://metr.org/notes/2026-01-22-time-horizon-limitations/. Retrieved
   2026-10-23.

5. Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, et al. (42 authors). "Measuring what
   Matters: Construct Validity in Large Language Model Benchmarks." arXiv:2511.04703, 2025
   (NeurIPS 2025 Datasets and Benchmarks track, per the arXiv comments field).
   https://arxiv.org/abs/2511.04703. Retrieved 2026-10-23.

6. Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail
   Yurochkin. "tinyBenchmarks: evaluating LLMs with fewer examples." arXiv:2402.14992, 2024
   (ICML 2024, per the arXiv comments field). https://arxiv.org/abs/2402.14992. Retrieved
   2026-10-23.

7. Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, et al. (85 authors). "Terminal-Bench:
   Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces." arXiv:2601.11868,
   2026. https://arxiv.org/abs/2601.11868. Retrieved 2026-10-23.

## Differences from the brief

- The brief's Anthropic URL (`/engineering/`) now redirects to `/research/`. Both URLs are in
  the record.
- The brief uses short titles for Miller ("Adding Error Bars to Evals"), "Signal and Noise", and
  "Measuring what Matters"; the full titles are above.
- The brief calls the METR note "Time horizon limitations"; its title is "Clarifying limitations
  of time horizon", and it is a research note by one author. The brief's "sensitivity to scoring
  choices" is only partly supported: the note discusses modeling and human-baseline choices, not
  grading rules.
- The brief's claims about "Measuring what Matters" (little statistical testing) and
  tinyBenchmarks (IRT) are not in the abstracts; both were confirmed in the arXiv HTML full text.

## Dropped

- Raj's raise-v1 instrument, https://github.com/raj200501/raise-v1-public. Not opened: this
  session's egress proxy returned HTTP 403 ("GitHub access to this repository is not enabled
  for this session"), and a request to attach the repository was denied by the permission
  system. It is listed under `dropped` in `artifacts/citecheck.json` (not under `references`),
  is not cited in README.md or docs/REPORT.md, and is under Needs Raj in STATUS.md for a
  manual check.
