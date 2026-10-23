import pytest

import evalstats
import noisefloor
from evalstats import cli


def test_versions_are_strings() -> None:
    assert isinstance(evalstats.__version__, str)
    assert isinstance(noisefloor.__version__, str)


@pytest.mark.parametrize("command", sorted(cli.SUBCOMMANDS))
def test_cli_subcommands_parse(command: str) -> None:
    args = cli.build_parser().parse_args([command])
    assert args.command == command


def test_cli_unimplemented_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compare"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
