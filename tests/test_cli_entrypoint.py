from pathlib import Path
import pytest
from core_score.cli import parser, main


@pytest.mark.parametrize("values", [
    ["p0.cif", "p1.pdb", "--out", "results"],
    ["--out", "results", "p0.cif", "p1.pdb"],
])
def test_direct_structures(values):
    args = parser().parse_args(values)
    assert args.inputs == [Path("p0.cif"), Path("p1.pdb")]
    assert not args.skip_posebusters and args.workers == 1


@pytest.mark.parametrize("name", ["inputs.csv", "inputs.CSV"])
def test_csv_input(name):
    args = parser().parse_args([name, "--out", "results", "--skip-posebusters"])
    assert args.inputs == [Path(name)]
    assert args.skip_posebusters


@pytest.mark.parametrize("words", [[], ["--help"]])
def test_help(words, capsys):
    with pytest.raises(SystemExit) as exc:
        main(words)
    assert exc.value.code == 0
    assert "core_score_input.csv" in capsys.readouterr().out


def test_python_run_loads_inputs_then_executes(tmp_path):
    from unittest.mock import patch
    import core_score

    with patch("core_score.pipeline.load_inputs", return_value=["groups"]) as load, \
            patch("core_score.pipeline.execute", return_value="summary") as execute:
        assert core_score.run("inputs.csv", tmp_path, ligand="CCO", workers=4) == "summary"
    assert load.call_args.args[0] == ["inputs.csv"]
    assert load.call_args.kwargs["ligand"] == "CCO"
    assert execute.call_args.args == (["groups"], tmp_path)
    assert execute.call_args.kwargs == dict(workers=4)


def test_public_python_interface_is_run_and_score():
    import core_score

    assert sorted(core_score.__all__) == ["run", "score"]
    assert core_score.score([[1.0] * 4] * 4) == 0.5
    assert not hasattr(core_score, "core_score")
