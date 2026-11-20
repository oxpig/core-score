"""Binding-site radius recovery in compare_pair, with OST replaced by canned results."""

from unittest.mock import patch
import pytest
from core_score.analysis.pairs import compare_pair


def scored(rmsd, lddt_pli=0.8):
    return dict(status="SUCCESS",
                rmsd=dict(assigned_scores=[dict(score=rmsd, bb_rmsd=0.1, lddt_lp=0.9)]),
                lddt_pli=dict(assigned_scores=[dict(score=lddt_pli)]))


def unassigned(reason):
    return dict(status="SUCCESS",
                rmsd=dict(assigned_scores=[], model_ligand_unassigned_reason={"B": reason}),
                lddt_pli=dict(assigned_scores=[dict(score=0.5)]))


BINDING_SITE = unassigned(["binding_site", "No residues in binding site"])


def compare(answers, radius_fallback=True, copies=1):
    calls = []

    def ost_call(model, reference, timeout, radius, lddt_pli):
        assert lddt_pli is False  # pose-versus-pose pairs never ask OST for lDDT-PLI
        calls.append(radius)
        return answers[radius]

    pose = lambda name: dict(pose_id=name, path=f"{name}.cif", prepared=dict(path=f"{name}.cif", copies=copies))
    with patch("core_score.analysis.ost.compare_ligands", side_effect=ost_call):
        row, detail = compare_pair(pose("a"), pose("b"), radius_fallback=radius_fallback)
    return row, detail, calls


def test_score_at_4_is_kept_without_retries():
    row, _, calls = compare({4.0: scored(1.234567890123)})
    assert calls == [4.0]
    assert row["rmsd"] == row["standard_rmsd"] == 1.234567890123
    assert row["binding_site_radius_angstrom"] == 4.0 and not row["radius_fallback_used"]


def test_first_success_stops_at_6():
    row, detail, calls = compare({4.0: BINDING_SITE, 6.0: scored(3.0)})
    assert calls == [4.0, 6.0]
    assert row["rmsd"] == 3.0 and row["binding_site_radius_angstrom"] == 6.0
    assert row["radius_fallback_used"] and row["standard_rmsd"] is None
    assert [a["radius"] for a in detail["radius_attempts"]] == [4.0, 6.0]


def test_8_only_after_6_failed():
    row, _, calls = compare({4.0: BINDING_SITE, 6.0: BINDING_SITE, 8.0: scored(2.0)})
    assert calls == [4.0, 6.0, 8.0]
    assert row["binding_site_radius_angstrom"] == 8.0


def test_graph_failure_is_not_retried():
    row, _, calls = compare({4.0: unassigned(["identity", "graph mismatch"])})
    assert calls == [4.0]
    assert not row["radius_fallback_used"] and row["error"]


def test_failed_retries_stay_missing():
    row, _, calls = compare({4.0: BINDING_SITE, 6.0: BINDING_SITE, 8.0: BINDING_SITE})
    assert calls == [4.0, 6.0, 8.0]
    assert row["rmsd"] is None and row["binding_site_radius_angstrom"] is None
    assert not row["radius_fallback_used"] and row["error"]


@pytest.mark.parametrize("radius_fallback, copies", [(False, 1), (True, 2)])
def test_no_retry_when_disabled_or_several_copies(radius_fallback, copies):
    _, _, calls = compare({4.0: BINDING_SITE}, radius_fallback=radius_fallback, copies=copies)
    assert calls == [4.0]


def test_recovered_pair_keeps_lddt_pli_from_4():
    row, detail, _ = compare({4.0: BINDING_SITE, 6.0: scored(2.0, lddt_pli=0.9)})
    assert row["rmsd"] == 2.0
    assert row["lddt_pli"] == 0.5
    assert len(detail["ost_results"]) == 2
