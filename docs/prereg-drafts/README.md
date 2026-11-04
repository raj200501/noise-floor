# Drafts of preregistrations 0004 to 0006 (not frozen)

These are the unfrozen drafts of `prereg/0004-e1-factorial.md`, `prereg/0005-e0-glmm-cells.md`,
and `prereg/0006-g6-recovery.md`, kept here so they survive a new container. Nothing here is
a preregistration: the frozen files will be assembled from these drafts and from freshly
generated blocks, and then frozen in one commit.

- `*.draft.md`: the texts, with placeholders (`LOCAL_TASKS_N`, `FREEZE_DATE`, ...) and a
  `## Blocks` list that `assemble3.py` fills from `tools/prereg_blocks.py` output.
- `assemble3.py`: fills the placeholders and blocks (`BLOCKS_DIR OUT_DIR FREEZE_DATE`).
- `freeze.sh`: the freeze sequence (fresh blocks, assembly, consistency check, llama-server
  pin assert, `tools/prereg.py freeze` for each file, prereg tests, `make prereg-verify`). Its
  paths point at the session scratchpad; point them here before running it.
- `fixes2.py` and `fixes3.py`: follow-ups from the verification of the final-review fixes
  (D-034), written but not applied yet (see STATUS.md). They expect the drafts under their
  scratchpad names (`0004-draft3.md` and so on).
