import json
from unittest.mock import Mock, patch
import numpy as np
import pytest
from core_score import score as core_score
from core_score.analysis.core import score_groups
from core_score.analysis.core import build_matrix
from core_score.cli import parser
from core_score.analysis.posebusters import aggregate
from core_score.analysis.posebusters import run_posebusters


def test_core_known_score_and_diagonal_independence():
    for diag in (0.0, float("nan"), 99.0):
        matrix = np.full((4, 4), 1.0)
        np.fill_diagonal(matrix, diag)
        assert core_score(matrix) == 0.5


def test_core_averages_each_pose_three_closest_other_poses_capped_at_threshold():
    # Four poses within 0.2 A of each other and one far away.
    matrix = np.full((5, 5), 10.0)
    matrix[:4, :4] = 0.2
    np.fill_diagonal(matrix, 0)
    # The four agree: three neighbours at 0.2 A, penalty 0.1 each. The fifth has none within T.
    assert core_score(matrix) == pytest.approx(1 - (4 * 0.1 + 1) / 5)
    assert core_score(matrix, threshold=1.0) == pytest.approx(1 - (4 * 0.2 + 1) / 5)
    # A pose with only one close partner is still penalised for the other two neighbours.
    pair = [[0, 0.1, 10, 10], [0.1, 0, 10, 10], [10, 10, 0, 10], [10, 10, 10, 0]]
    assert core_score(pair) == pytest.approx(1 - (2 * (0.05 + 1 + 1) / 3 + 2) / 4)
    assert core_score(pair, neighbours=1) == pytest.approx(1 - (0.05 + 0.05 + 1 + 1) / 4)


def test_core_needs_four_poses_and_never_uses_fewer_neighbours():
    with pytest.raises(ValueError, match="at least 4 poses"):
        core_score([[0, 0.1, 10], [0.1, 0, 10], [10, 10, 0]])


def test_core_endpoints():
    assert core_score(np.zeros((4, 4))) == 1.0
    assert core_score(np.full((4, 4), 2.0)) == 0.0  # Exactly T apart counts as unsupported.
    with pytest.raises(ValueError):
        core_score([[0]])


def test_scores_at_each_threshold_and_never_from_incomplete_matrices(tmp_path):
    poses = [dict(pose_id=p) for p in "abcd"]
    rows = [dict(ensemble="g", site="1", pose_i=a, pose_j=b, rmsd=r, n_assigned=1, error="")
            for a, b, r in [("a", "b", 0.5), ("a", "c", 3.0), ("b", "c", 1.0),
                            ("a", "d", 3.0), ("b", "d", 3.0), ("c", "d", 3.0)]]
    groups = [dict(ensemble="g", site="1", name="g", poses=poses),
              dict(ensemble="h", site="1", name="h", poses=poses),
              dict(ensemble="u", site="unresolved", name="u", poses=poses, error=dict(error="ValueError: no sites"))]
    scores = score_groups(groups, rows, thresholds=(1.0, 2.0))
    assert [(s["ensemble"], s["threshold_angstrom"]) for s in scores] == [
        ("g", 1.0), ("g", 2.0), ("h", 1.0), ("h", 2.0), ("u", 1.0), ("u", 2.0)]
    # T = 2 A, three other poses each: a (0.5, 3, 3), b (0.5, 1, 3), c (1, 3, 3), d (3, 3, 3).
    penalties = [(0.25 + 1 + 1) / 3, (0.25 + 0.5 + 1) / 3, (0.5 + 1 + 1) / 3, 1]
    assert scores[1]["core_score"] == pytest.approx(1 - sum(penalties) / 4)
    assert all(s["core_score"] is None and s["error"] for s in scores[2:])


def test_matrix_missing_duplicate_and_failed_pairs():
    row = dict(pose_i="a", pose_j="b", rmsd=1, n_assigned=1, error="")
    np.testing.assert_array_equal(build_matrix(["a", "b"], [row]), [[0, 1], [1, 0]])
    for rows in ([], [dict(row, error="OST failed")], [dict(row, n_assigned=2)]):
        with pytest.raises(ValueError):
            build_matrix(["a", "b"], rows)


def test_cli_agreed_defaults():
    args = parser().parse_args(["one.cif", "two.cif", "--out", "results"])
    assert args.workers == 1 and args.timeout is None
    assert not args.skip_posebusters and not hasattr(args, "filter_pb_invalid")
    assert not args.no_radius_fallback and not hasattr(args, "fallback_radii")
    assert args.rmsd_thresholds == [2]
    assert not hasattr(args, "site_mode")


def test_pb_unknown_never_passes():
    assert aggregate([True, None]) is None
    assert aggregate([True, False, None]) is False
    assert aggregate([]) is None


def test_posebusters_records_every_pose(tmp_path):
    groups = [dict(ensemble="g", poses=[dict(ensemble="g", pose_id="a"), dict(ensemble="g", pose_id="b")])]
    answers = [dict(ensemble="g", pose_id="a", pb_valid=True), dict(ensemble="g", pose_id="b", pb_valid=False)]
    with patch("posebusters.PoseBusters", return_value=Mock(config={"modules": []})), \
            patch("core_score.analysis.posebusters.check_pose", side_effect=answers) as pb:
        records = run_posebusters(groups, tmp_path)
    assert pb.call_count == 2
    assert [r["pb_valid"] for r in records] == [True, False]
    assert json.loads((tmp_path / "posebusters.json").read_text()) == answers


def test_posebusters_energy_threads_are_bounded_without_mutating_shared_config(tmp_path):
    original = {"modules": [{"function": "energy_ratio", "parameters": {"num_threads": 0, "threshold": 7}}]}
    buster = Mock(config=original)
    with patch("posebusters.PoseBusters", return_value=buster):
        run_posebusters([], tmp_path)
    assert original["modules"][0]["parameters"] == {"num_threads": 0, "threshold": 7}
    assert buster.config["modules"][0]["parameters"] == {"num_threads": 1, "threshold": 7}


def test_missing_energy_is_reported_as_unknown_in_csv_and_never_counted_valid(tmp_path):
    from core_score.analysis.posebusters import check_pose
    from core_score.outputs.report import write_report

    sites, scores, pb, references = report_inputs()
    pose = dict(ensemble="g", pose_id="a", prepared=dict(receptor="r.pdb", copy_records=["c0"]))
    copy = dict(copy_id="0", pb_valid=None, checks=dict(bond_lengths=True, internal_energy=None),
                tests=dict(energy_ratio=None, ensemble_avg_energy=None))
    with patch("core_score.analysis.posebusters.check_copy", return_value=copy):
        unknown = check_pose(None, ["bond_lengths", "internal_energy"], pose)
    pb[0] = unknown
    write_report(sites, scores, pb, references, tmp_path)
    row = read(tmp_path / "predictions.csv")[0]
    assert row["pb_valid"] == ""
    assert row["core_score"] == "0.9"
    assert "internal_energy" in row["note"]
    assert read(tmp_path / "core_scores.csv")[0]["valid_prediction_count"] == "0"


def test_each_check_combines_the_copies_and_unknown_never_passes():
    from core_score.analysis.posebusters import check_pose

    pose = dict(ensemble="g", pose_id="a", prepared=dict(receptor="r.pdb", copy_records=["c0", "c1"]))
    copies = [dict(copy_id="0", pb_valid=False, checks=dict(bond_lengths=True, internal_energy=False)),
              dict(copy_id="1", pb_valid=None, error="ValueError: x")]
    with patch("core_score.analysis.posebusters.check_copy", side_effect=copies):
        record = check_pose(None, ["bond_lengths", "internal_energy"], pose)
    assert record["pb_valid"] is False
    assert record["checks"] == dict(bond_lengths=None, internal_energy=False)
    unprepared = check_pose(None, ["bond_lengths"], dict(ensemble="g", pose_id="b", error="bad input"))
    assert unprepared["checks"] == dict(bond_lengths=None)


def test_pipeline_scores_every_pose_and_can_skip_posebusters(tmp_path):
    from core_score.pipeline import execute

    stages = ["require_ost", "prepare_groups", "prepare_site_groups", "compare_groups", "score_groups"]
    patches = [patch(f"core_score.pipeline.{name}", return_value=[]) for name in stages]
    for p in patches:
        p.start()
    try:
        with patch("core_score.pipeline.run_posebusters") as pb:
            execute([], tmp_path / "default")
            assert pb.call_count == 1
            execute([], tmp_path / "skip", skip_posebusters=True)
            assert pb.call_count == 1
    finally:
        for p in patches:
            p.stop()


def test_reference_results_are_raw_scores_for_every_pose(tmp_path):
    from core_score.analysis.reference import evaluate_groups

    pose = dict(ensemble="g", pose_id="p", path="pose.cif", prepared=dict(copies=1))
    reference = dict(ensemble="g", pose_id="reference", path="ref.cif")
    groups = [dict(ensemble="g", poses=[pose], reference=reference), dict(ensemble="h", poses=[pose])]
    row = dict(ensemble="g", site="reference", pose_i="p", pose_j="reference", rmsd=1.0, lddt_pli=0.7, error="")
    with patch("core_score.analysis.reference.prepare", return_value=dict(reference, prepared=dict(copies=1))), \
            patch("core_score.analysis.pairs.pair_job", return_value=(row, {})) as compare:
        results = evaluate_groups(groups, tmp_path)
    assert compare.call_count == 1  # h has no reference
    assert results == [dict(ensemble="g", pose_id="p", rmsd=1.0, lddt_pli=0.7, error=None)]


def test_failed_reference_comparison_is_blank_with_its_reason():
    from core_score.analysis.reference import reference_result

    result = reference_result(dict(ensemble="g", pose_id="p", rmsd=0.5, lddt_pli=0.9, error="ambiguous"))
    assert (result["rmsd"], result["lddt_pli"], result["error"]) == (None, None, "ambiguous")
    assert reference_result(dict(ensemble="g", pose_id="p", error=dict(error="ValueError: x")))["error"] == "ValueError: x"


def report_inputs():
    """Ensembles h and g score; u could not be split into sites. g has a reference."""
    sites = [dict(ensemble=e, site=s, name=e, poses=[dict(ensemble=e, pose_id=p, confidence=c) for p, c in poses])
             for e, s, poses in (("h", "1", [("a", 0.9)]), ("g", "1", [("a", 0.8), ("b", 0.6)]),
                                 ("u", "unresolved", [("a", 0.9)]))]
    scores = [dict(ensemble="h", site="1", name="h", threshold_angstrom=2.0, core_score=0.6, error=None),
              dict(ensemble="g", site="1", name="g", threshold_angstrom=2.0, core_score=0.9, error=None),
              dict(ensemble="u", site="unresolved", name="u", threshold_angstrom=2.0, core_score=None,
                   error="ValueError: x")]
    pb = [dict(ensemble=e, pose_id=p, pb_valid=v) for e, p, v in
          (("g", "a", True), ("g", "b", False), ("h", "a", True), ("u", "a", True))]
    references = [dict(ensemble="g", pose_id="a", rmsd=0.4, lddt_pli=0.9, error=None),
                  dict(ensemble="g", pose_id="b", rmsd=None, lddt_pli=None, error="ambiguous")]
    return sites, scores, pb, references


def read(path):
    import csv
    return list(csv.DictReader(path.open()))


def test_core_scores_has_one_row_per_complex_best_first(tmp_path):
    from core_score.outputs.report import write_report

    sites, scores, pb, references = report_inputs()
    assert write_report(sites, scores, pb, references, tmp_path) == {2.0: [0.6, 0.9, None]}
    rows = read(tmp_path / "core_scores.csv")
    assert list(rows[0]) == ["ensemble", "core_score", "prediction_count", "valid_prediction_count", "note"]
    assert [(r["ensemble"], r["core_score"], r["prediction_count"], r["valid_prediction_count"]) for r in rows] == [
        ("g", "0.9", "2", "1"), ("h", "0.6", "1", "1"), ("u", "", "1", "1")]
    assert rows[2]["note"] == "ValueError: x"


def test_predictions_has_every_prediction_with_its_complex_core_score(tmp_path):
    from core_score.outputs.report import write_report

    sites, scores, pb, references = report_inputs()
    sites[2]["poses"][0]["error"] = dict(error="ValueError: no ligand")
    write_report(sites, scores, pb, references, tmp_path)
    rows = read(tmp_path / "predictions.csv")
    assert list(rows[0]) == ["ensemble", "prediction", "core_score", "confidence", "pb_valid", "rmsd", "lddt_pli", "note"]
    assert [(r["ensemble"], r["prediction"], r["core_score"]) for r in rows] == [
        ("g", "a", "0.9"), ("g", "b", "0.9"), ("h", "a", "0.6"), ("u", "a", "")]
    assert (rows[0]["pb_valid"], rows[0]["rmsd"], rows[0]["lddt_pli"]) == ("True", "0.4", "0.9")
    assert (rows[1]["rmsd"], rows[1]["note"]) == ("", "ambiguous")
    assert rows[3]["note"] == "ValueError: no ligand"


def test_reports_leave_out_what_was_not_given(tmp_path):
    from core_score.outputs.report import write_report

    sites, scores, _, _ = report_inputs()
    for site in sites:
        for pose in site["poses"]:
            del pose["confidence"]
    scores[2] = dict(scores[2], core_score=0.1, error=None)
    write_report(sites, scores, None, [], tmp_path)
    assert list(read(tmp_path / "core_scores.csv")[0]) == ["ensemble", "core_score", "prediction_count"]
    assert list(read(tmp_path / "predictions.csv")[0]) == ["ensemble", "prediction", "core_score"]


def test_several_thresholds_give_a_core_column_each_and_sites_keep_their_names(tmp_path):
    from core_score.outputs.report import write_report

    sites = [dict(ensemble="g", site=s, name=f"g_site_{s}", poses=[dict(ensemble="g", pose_id="a")]) for s in "12"]
    scores = [dict(ensemble="g", site=s, name=f"g_site_{s}", threshold_angstrom=t, core_score=c, error=None)
              for s, cores in (("1", (0.5, 0.9)), ("2", (0.2, 0.4))) for t, c in zip((1.0, 2.0), cores)]
    write_report(sites, scores, None, [], tmp_path)
    rows = read(tmp_path / "core_scores.csv")
    assert [(r["ensemble"], r["core_score_1A"], r["core_score_2A"]) for r in rows] == [
        ("g_site_1", "0.5", "0.9"), ("g_site_2", "0.2", "0.4")]


def test_plot_dots_shrink_as_the_benchmark_grows():
    from core_score.outputs.plot import dot_size

    assert dot_size(6) > dot_size(900) > dot_size(7000)


def test_plot_writes_both_pngs_per_threshold(tmp_path):
    pytest.importorskip("matplotlib")
    from core_score.outputs.plot import plot_core_scores

    plot_core_scores({1.0: [0.4, 0.9, None], 2.0: [0.5, 0.95, None]}, tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "core_score_distribution_1A.png", "core_score_distribution_2A.png", "core_scores_1A.png", "core_scores_2A.png"]


def test_missing_openstructure_says_how_to_install_it(monkeypatch):
    from core_score.analysis.ost import require_ost

    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="conda install -c conda-forge -c bioconda openstructure"):
        require_ost()
