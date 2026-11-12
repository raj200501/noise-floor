# noise-floor make targets (docs/BUILD_BRIEF.md section 10.1 step 7).
# Targets not built yet call tools/todo.py, which fails loudly and names its gate.

UV     ?= uv
RUN    := $(UV) run
TOOL   := $(UV) run --no-project python
TODO   := $(TOOL) tools/todo.py
PKGS   := src/evalstats src/noisefloor

.PHONY: setup doctor test lint typecheck ci demo prereg-verify claimcheck freshness \
        citecheck secrets-scan budget review report reproduce ship-check prepush apidoc

# The R extra (rpy2 for lme4) is installed only where R is present.
R_EXTRA := $(if $(shell command -v Rscript 2>/dev/null),--extra r,)

setup:
	$(UV) sync --locked $(R_EXTRA)

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
	$(TOOL) tools/prereg.py verify

claimcheck:
	$(TOOL) tools/claimcheck.py README.md docs/REPORT.md

freshness:
	$(TOOL) tools/freshness.py README.md docs/REPORT.md

citecheck:
	$(TOOL) tools/citecheck.py

secrets-scan:
	$(TOOL) tools/secrets_scan.py

budget:
	$(TOOL) tools/budget.py

review:
	$(TOOL) tools/review.py

report:
	$(RUN) python -m noisefloor.report

reproduce:
	$(RUN) python -m noisefloor.reanalysis --verify
	$(RUN) python -m noisefloor.report
	$(RUN) python tools/noise_floor.py --check
	$(RUN) python tools/validity_summary.py --check
	$(RUN) python -m noisefloor.reanalysis report
	git diff --exit-code --stat -- docs/figures docs/leaderboards
	@echo "REPRODUCE: PASS (E2 and E3 recomputed from the pinned snapshots; figures unchanged)"

ship-check:
	$(TOOL) tools/ship_check.py

# Fresh-clone make setup + make ci before every push (D-015).
prepush:
	$(TOOL) tools/prepush.py

# Regenerate docs/api/evalstats.md from docstrings.
apidoc:
	$(RUN) python tools/apidoc.py
