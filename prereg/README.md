# Preregistrations

Files named `NNNN-<slug>.md`, committed and hashed before any confirmatory data exists. A frozen
file is never edited. A needed change is a new dated amendment file that cites the original, is
committed before any data it affects, and is reported as a deviation.

Machine-readable parts live in fenced blocks that tools/ship_check.py reads, for example
` ```json e0-grid `, ` ```json g6-seeds `, and ` ```json e1-design ` (see docs/GATES.md).
