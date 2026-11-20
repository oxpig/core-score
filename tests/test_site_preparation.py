from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
from core_score.inputs.structures import read_document, ligand_residues
from core_score.inputs.sites import prepare_site_groups


@pytest.mark.parametrize('native', [False, True])
def test_multicopy_inputs_keep_corresponding_sites_and_original_coordinates(tmp_path, native):
    poses = []
    originals = {}
    for index, positions in enumerate(([-10, 10], [10, -10])):
        source = tmp_path / f'p{index}.cif'
        source.write_text('''data_pose
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C CA A 0 0 0
''' + f'HETATM 2 C C1 B {positions[0]} 0 0\nHETATM 3 C C1 C {positions[1]} 0 0\n')
        originals[source] = source.read_bytes()
        keys = list(ligand_residues(read_document(source)))
        records = []
        for i, x in enumerate(positions):
            record = dict(copy_id=str(i), centre=[x, 0, 0], ost_sdf=str(tmp_path / f'p{index}_{i}.sdf'))
            if native:
                record['residue'] = list(keys[i])
            records.append(record)
        sdfs = [] if native else [r['ost_sdf'] for r in records]
        poses.append(dict(ensemble='target', pose_id=f'p{index}',
                          prepared=dict(path=str(source), copies=2, copy_records=records, ligand_sdfs=sdfs)))
    groups = [dict(ensemble='target', poses=poses)]
    original_groups = deepcopy(groups)
    fit = dict(transform=np.eye(4), rmsd=0, chain_mapping={})
    with patch('core_score.analysis.ost.superpose', return_value=fit) as align:
        sites = prepare_site_groups(groups, tmp_path)
    assert align.call_count == 1
    assert [site['name'] for site in sites] == ['target_site_1', 'target_site_2']
    for site in sites:
        prepared = [p['prepared'] for p in site['poses']]
        assert all(p['copies'] == 1 for p in prepared)
        copies = [p['copy_records'][0] for p in prepared]
        assert copies[0]['centre'] == copies[1]['centre']
        assert copies[0]['copy_id'] != copies[1]['copy_id']
        for result, copy in zip(prepared, copies):
            if native:
                selected = ligand_residues(read_document(result['path']))
                assert list(selected) == [tuple(copy['residue'])]
                assert float(next(iter(selected.values()))[0]['Cartn_x']) == copy['centre'][0]
            else:
                assert result['ligand_sdfs'] == [copy['ost_sdf']]
    assert groups == original_groups
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_single_copy_does_not_align(tmp_path):
    groups = [dict(ensemble='target', poses=[dict(prepared=dict(copies=1))])]
    with patch('core_score.analysis.ost.superpose') as align:
        result = prepare_site_groups(groups, tmp_path)
    align.assert_not_called()
    assert result[0]['site'] == '1'
