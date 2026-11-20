import logging
from types import SimpleNamespace
from core_score.analysis.posebusters import check_copy


def test_unknown_energy_retains_reason_and_other_checks():
    logger = logging.getLogger('posebusters.modules.energy_ratio')
    original_handlers = list(logging.getLogger('posebusters').handlers)
    class Buster:
        def bust(self, *args, **kwargs):
            logger.warning('UFF parameters missing for molecule.')
            return SimpleNamespace(iloc=[{'internal_energy': float('nan'), 'bond_lengths': True}])
    result = check_copy(Buster(), ['internal_energy', 'bond_lengths'],
                        dict(copy_id='0', provenance='native', pb_error=None, pb_sdf='a.sdf'), 'r.pdb')
    assert result['pb_valid'] is None
    assert result['checks']['bond_lengths'] is True
    assert result['error'] == 'UFF parameters missing for molecule.'
    assert logging.getLogger('posebusters').handlers == original_handlers


def test_diagnostic_handler_removed_on_exception():
    original_handlers = list(logging.getLogger('posebusters').handlers)
    class Buster:
        def bust(self, *args, **kwargs): raise ValueError('bad input')
    result = check_copy(Buster(), [], dict(copy_id='0', provenance='native', pb_error=None, pb_sdf='a.sdf'), 'r.pdb')
    assert result['pb_valid'] is None
    assert logging.getLogger('posebusters').handlers == original_handlers
