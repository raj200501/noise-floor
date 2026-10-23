# Security

## Reporting

Please report security problems privately to the repository owner through GitHub's private
vulnerability reporting for this repository. Do not open a public issue for a security problem.

## Scope and design

- noise-floor runs model agents only inside disposable sandboxes: Docker with `--network none`
  when available, otherwise an OS sandbox that blocks network access and confines writes to a
  temporary workspace. Tools given to agents are mock tools.
- No real credentials or real systems are used in any experiment. Secrets stay in environment
  variables and are never written to files, logs, artifacts, or command lines.
  `make secrets-scan` checks every tracked file and every blob in git history.
- The project measures evals. It does not build anything that helps an agent evade graders,
  monitors, or safety measures beyond what the measurement strictly requires.
