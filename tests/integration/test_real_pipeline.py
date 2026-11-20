"""Small real-structure regression independent of the external seven-model archive.

Run explicitly with OST on PATH, or CORE_SCORE_TEST_OST set to its executable.
Missing OST or changed frozen inputs are errors, not skipped tests.

OST calculates in single precision, so different builds of the same version give answers
a few float32 steps apart. Frozen values are compared to a relative 1e-5.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import pytest
from core_score.pipeline import execute
from core_score.inputs.inputs import load_inputs

FIXTURES = Path(__file__).parents[1] / "fixtures"
TOLERANCE = 1e-5


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, checksum in manifest["structures"].items():
        assert file_hash(FIXTURES / name) == checksum
    return manifest


def options(monkeypatch):
    # Pin the OST version the values were frozen with; production has no preflight check.
    import subprocess

    executable = shutil.which(os.environ.get("CORE_SCORE_TEST_OST", "ost"))
    assert executable, "OST is required for the real-structure regression"
    version = subprocess.check_output([executable, "--version"], text=True)
    assert inputs()["ost_version"] in version
    # The package runs `ost` from PATH.
    monkeypatch.setenv("PATH", str(Path(executable).parent) + os.pathsep + os.environ["PATH"])


def groups(folder):
    """Four poses, the two frozen structures twice each, since CORE needs at least four."""
    folder.mkdir()
    for name, source in zip("abcd", ("7wl4_pose0.cif", "7wl4_pose1.cif") * 2):
        shutil.copy(FIXTURES / source, folder / f"{name}.cif")
    return load_inputs([folder / f"{name}.cif" for name in "abcd"], ligand_chains=["B"], ligand=inputs()["smiles"])


def test_original_ost_values_and_parallel_exports(tmp_path, monkeypatch):
    # Exercise OST's own library discovery, without a package-supplied path.
    monkeypatch.delenv("OST_COMPOUNDS_CHEMLIB", raising=False)
    monkeypatch.delenv("OST_ROOT", raising=False)
    options(monkeypatch)
    expected = inputs()["expected_single_copy"]
    outputs = []
    poses = groups(tmp_path / "poses")
    for workers in (1, 2):
        summary = execute(
            poses,
            tmp_path / str(workers),
            workers=workers,
            skip_posebusters=True,
        )
        with (tmp_path / str(workers) / "core_scores.csv").open() as handle:
            scores = list(csv.DictReader(handle))
        with (tmp_path / str(workers) / "details" / "pairs.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert summary["score_count"] == 1
        assert summary["completed_comparisons"] == 6
        # Each pose's three closest others are its copy (0 A) and the two other poses (RMSD each).
        assert float(scores[0]["core_score"]) == pytest.approx(1 - (2 * expected["rmsd"] / 2) / 3, rel=TOLERANCE)
        for field, value in expected.items():
            if isinstance(value, (int, float)):
                assert float(rows[0][field]) == pytest.approx(value, rel=TOLERANCE), field
            else:
                assert rows[0][field] == value, field
        outputs.append(rows)
    assert outputs[0] == outputs[1]


def test_pb_records_invalid_real_poses_and_core_still_uses_them(tmp_path, monkeypatch):
    options(monkeypatch)
    summary = execute(groups(tmp_path / "poses"), tmp_path / "pb")
    pb = json.loads((tmp_path / "pb/details/posebusters.json").read_text())
    assert len(pb) == 4
    assert all(p["pb_valid"] is False for p in pb)
    with (tmp_path / "pb/core_scores.csv").open() as handle:
        scores = list(csv.DictReader(handle))
    assert summary["comparison_count"] == 6
    assert summary["score_count"] == 1
    assert scores[0]["prediction_count"] == "4"
    assert scores[0]["core_score"] is not None


def test_two_ligand_copies_are_scored_as_two_sites(tmp_path, monkeypatch):
    options(monkeypatch)
    folder = tmp_path / "poses"
    groups(folder)
    both_copies = load_inputs([folder / f"{name}.cif" for name in "abcd"], ligand=inputs()["smiles"])
    execute(both_copies, tmp_path / "sites", skip_posebusters=True)
    with (tmp_path / "sites" / "core_scores.csv").open() as handle:
        scores = {row["ensemble"]: float(row["core_score"]) for row in csv.DictReader(handle)}
    # Copy B moves 0.28 A between the two structures, so CORE = 1 - RMSD / 3 as above.
    # Copy C moves more than 2 A, so each pose's only close neighbour is its own copy: CORE = 1/3.
    expected = inputs()["expected_single_copy"]["rmsd"]
    assert scores == pytest.approx({"ensemble_site_1": 1 - expected / 3, "ensemble_site_2": 1 / 3}, rel=TOLERANCE)


def test_command_line_run_with_reference_and_plots(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from core_score.cli import main

    options(monkeypatch)
    groups(tmp_path / "poses")
    shutil.copy(FIXTURES / "7wl4_pose0.cif", tmp_path / "reference.cif")
    table = tmp_path / "core_score_input.csv"
    table.write_text("ensemble,ligand,predictions,reference,ligand_chain\n"
                     f"7wl4,{inputs()['smiles']},poses/*.cif,reference.cif,B\n")
    assert main([str(table), "--out", str(tmp_path / "results"), "--workers", "2", "--plot"]) == 0
    results = tmp_path / "results"
    assert sorted(p.name for p in results.iterdir()) == [
        "core_score_distribution.png", "core_scores.csv", "core_scores.png", "details", "predictions.csv"]
    # The evidence is kept, the working copies of every structure are not.
    assert sorted(p.name for p in (results / "details").iterdir()) == [
        "ligand_preparation.json", "pairs.csv", "posebusters.json", "reference_details.json", "run.json"]
    with (results / "predictions.csv").open() as handle:
        rows = {row["prediction"]: row for row in csv.DictReader(handle)}
    # a and c are copies of the reference itself.
    for name in ("poses/a.cif", "poses/c.cif"):
        assert float(rows[name]["rmsd"]) == pytest.approx(0, abs=1e-3)
        assert float(rows[name]["lddt_pli"]) == pytest.approx(1)


def test_persistent_cli_matches_fresh_cli_after_different_inputs_and_options(tmp_path, monkeypatch):
    from core_score.analysis import ost, transport
    from core_score.inputs.prepare import prepare_groups

    options(monkeypatch)
    [group] = prepare_groups(groups(tmp_path / "poses"), tmp_path / "prepared")
    a, b = [pose["prepared"] for pose in group["poses"][:2]]
    cases = [(a, b, 4.0, True), (b, a, 6.0, False), (a, a, 4.0, True), (a, b, 4.0, True)]
    ordinary = []
    monkeypatch.setenv("CORE_SCORE_OST_MODE", "subprocess")
    for model, reference, radius, lddt in cases:
        result = ost.compare_ligands(model, reference, 30, radius, lddt)
        assert result["status"] == "SUCCESS"
        result.pop("_execution")
        result["arguments"].pop("output")  # Each call deliberately has a fresh output path.
        ordinary.append(result)
    monkeypatch.setenv("CORE_SCORE_OST_MODE", "persistent")
    pids = []
    with transport.session(tmp_path / "audit"):
        for case, expected in zip(cases, ordinary):
            model, reference, radius, lddt = case
            result = ost.compare_ligands(model, reference, 30, radius, lddt)
            execution = result.pop("_execution")
            result["arguments"].pop("output")
            assert execution["transport"] == "persistent_original_cli"
            pids.append(execution["worker_pid"])
            assert result == expected
    assert len(set(pids)) == 1
