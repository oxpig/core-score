"""Recover missing CIF metadata and explicit symmetry limits without masking other errors."""
import copy
import json
from unittest.mock import patch
import numpy as np
import pytest
from core_score.analysis import ost


def reply(reason=None):
    return dict(status="SUCCESS", rmsd=dict(
        assigned_scores=[] if reason else [dict(score=0.3)],
        model_ligand_unassigned_reason={"ligand": [reason, "diagnostic"]} if reason else {}))


def run_replies(results):
    calls = []
    def execute(args, timeout):
        calls.append(args)
        return copy.deepcopy(results[len(calls)-1]), dict(returncode=0, stdout="", stderr="")
    pose=dict(path="pose.cif", ligand_sdfs=["verified.sdf"])
    with patch.object(ost, "execute", execute), patch.object(ost.transport, "audit") as audit:
        result=ost.compare_ligands(pose, pose, 12)
    return result, calls, audit


def test_symmetry_retry_preserves_input_and_stops_after_success():
    result,calls,audit=run_replies([reply("symmetries"),reply()])
    assert len(calls)==2
    assert calls[1]==calls[0]+["--max-symmetries","100000"]
    assert result['rmsd']['assigned_scores'][0]['score']==0.3
    assert len(result['_execution']['symmetry_attempts'])==2
    json.dumps(result)  # Nested execution evidence must stay serializable.
    audit.assert_called_once()


def test_symmetry_retry_is_bounded_and_keeps_unknown_result():
    result,calls,_=run_replies([reply("symmetries")]*3)
    assert len(calls)==3 and calls[-1][-1]=='1000000'
    assert not result['rmsd']['assigned_scores']


@pytest.mark.parametrize('reason',[None,'identity','target_binding_site','model_binding_site'])
def test_other_results_do_not_trigger_symmetry_retry(reason):
    result,calls,audit=run_replies([reply(reason)])
    assert len(calls)==1
    assert 'symmetry_attempts' not in result['_execution']
    audit.assert_not_called()


def test_alignment_requests_fault_tolerant_metadata_reader():
    def execute(args,timeout):
        assert '--fault-tolerant' in args and '--rigid-scores' in args
        return dict(status='SUCCESS',transform=np.eye(4).tolist(),rmsd=0.2), dict(returncode=0)
    with patch.object(ost,'execute',execute):
        result=ost.superpose('a.cif','b.cif',10)
    np.testing.assert_array_equal(result['transform'],np.eye(4))


def test_alignment_still_rejects_unusable_transform():
    with patch.object(ost,'execute',return_value=(dict(status='SUCCESS',transform=None),dict(returncode=0))):
        with pytest.raises(ost.OSTError):
            ost.superpose('a.cif','b.cif',10)
