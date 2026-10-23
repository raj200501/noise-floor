# noise-floor make targets (docs/BUILD_BRIEF.md section 10.1 step 7).
# Targets not built yet call tools/todo.py, which fails loudly and names its gate.

UV     ?= uv
RUN    := $(UV) run
TOOL   := $(UV) run --no-project python
TODO   := $(TOOL) tools/todo.py
PKGS   := src/evalstats src/noisefloor

.PHONY: setup doctor test lint typecheck ci demo prereg-verify claimcheck freshness \
        citecheck secrets-scan budget review report reproduce ship-check

setup:
	$(UV) sync --locked

doctor:
	$(TOOL) tools/doctor.py

test:
	$(RUN) pytest --cov --cov-report=term-missing

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:
	$(RUN) mypy --strict $(PKGS)

ci: lint typecheck test demo

demo:
	$(RUN) python -m noisefloor.demo

prereg-verify:
	$(TODO) prereg-verify G11

claimcheck:
	$(TODO) claimcheck G10 README.md docs/REPORT.md

freshness:
	$(TODO) freshness G10 README.md docs/REPORT.md

citecheck:
	$(TODO) citecheck G13

secrets-scan:
	$(TOOL) tools/secrets_scan.py

budget:
	$(TOOL) tools/budget.py

review:
	$(TOOL) tools/review.py

report:
	$(TODO) report G13

reproduce:
	$(TODO) reproduce G13

ship-check:
	$(TOOL) tools/ship_check.py
