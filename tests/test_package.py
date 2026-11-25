import pytest

import evalstats
import noisefloor
from evalstats import cli


def test_versions_and_exports() -> None:
    assert isinstance(evalstats.__version__, str)
    assert isinstance(noisefloor.__version__, str)
    for name in evalstats.__all__:
        assert hasattr(evalstats, name)


def test_every_cli_subcommand_runs() -> None:
    # The CLI lists no command it cannot run (`report` was once listed and exited "not
    # implemented"); each listed subcommand has a handler.
    assert set(cli.SUBCOMMANDS) == set(cli.COMMANDS)


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
