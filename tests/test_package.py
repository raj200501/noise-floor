import pytest

import evalstats
import noisefloor
from evalstats import cli


def test_versions_and_exports() -> None:
    assert isinstance(evalstats.__version__, str)
    assert isinstance(noisefloor.__version__, str)
    for name in evalstats.__all__:
        assert hasattr(evalstats, name)


@pytest.mark.parametrize("command", ["rank", "report"])
def test_cli_unimplemented_exits_nonzero(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([command]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
