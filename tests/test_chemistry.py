import gemmi
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from core_score.inputs.structures import (
    matched_template,
    generic_graph,
)
from core_score.inputs.structures import canonical_smiles


def conformer(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=9)
    return Chem.RemoveAllHs(mol)


def test_transfer_preserves_chemistry_coordinates_and_atom_order():
    template = Chem.MolFromSmiles("CC(=O)[O-]")
    original = conformer("CC(=O)[O-]")
    observed = Chem.RenumberAtoms(generic_graph(original), [3, 1, 2, 0])
    result = matched_template(template, observed)
    Chem.SanitizeMol(result)
    assert canonical_smiles(result) == canonical_smiles(template)
    np.testing.assert_array_equal(
        result.GetConformer().GetPositions(), observed.GetConformer().GetPositions()
    )
    assert [a.GetSymbol() for a in result.GetAtoms()] == [
        a.GetSymbol() for a in observed.GetAtoms()
    ]


def test_connectivity_mismatch_is_not_repaired():
    observed = conformer("CCC")
    with pytest.raises(ValueError, match="identity"):
        matched_template(Chem.MolFromSmiles("C1CC1"), observed)


def test_native_bonds_select_correct_isomer_without_dropping_unknown_graph(monkeypatch):
    from core_score.inputs import structures as s
    molecules = {"A": conformer("CCO"), "B": conformer("COC"), "C": conformer("CCO")}
    residues = {(chain, "1", "", "LIG"): [dict(type_symbol=a.GetSymbol(), label_atom_id=str(i),
                  chain=chain) for i, a in enumerate(mol.GetAtoms())]
                for chain, mol in molecules.items()}
    monkeypatch.setattr(s, "ligand_residues", lambda doc: residues)
    monkeypatch.setattr(s, "native_graph", lambda doc, rows:
                        (None, None) if rows[0]["chain"] == "C" else
                        (molecules[rows[0]["chain"]], "native_cif_bonds"))
    assert {k[0] for k in s.choose_ligands(None, dict(smiles="CCO"))} == {"A", "C"}
    # An explicit chain request must not silently select another molecule.
    assert {k[0] for k in s.choose_ligands(None, dict(smiles="CCO", ligand_chains=["B"]))} == {"B"}


def test_native_mismatch_without_a_match_remains_a_preparation_error(monkeypatch):
    from core_score.inputs import structures as s
    mol = conformer("COC")
    residues = {(chain, "1", "", "LIG"): [dict(type_symbol=a.GetSymbol(), label_atom_id=str(i))
                for i, a in enumerate(mol.GetAtoms())] for chain in ("A", "B")}
    monkeypatch.setattr(s, "ligand_residues", lambda doc: residues)
    monkeypatch.setattr(s, "native_graph", lambda doc, rows: (mol, "native_cif_bonds"))
    assert len(s.choose_ligands(None, dict(smiles="CCO"))) == 2


def test_failed_chemistry_validation_cannot_return_pb_input(monkeypatch):
    import gemmi
    from core_score.inputs import structures as sdf

    # An otherwise valid molecule must be discarded if a later validation fails.
    observed = conformer("CCO")
    rows = [dict(type_symbol=a.GetSymbol(), label_atom_id=f"a{i}",
                 Cartn_x=str(x), Cartn_y=str(y), Cartn_z=str(z))
            for i, (a, (x, y, z)) in enumerate(zip(
                observed.GetAtoms(), observed.GetConformer().GetPositions()))]
    monkeypatch.setattr(sdf, "native_graph", lambda *args: (observed, "native_cif_bonds"))
    def reject(mol):
        raise ValueError("Synthetic sanitization failure")
    monkeypatch.setattr(sdf.Chem, "SanitizeMol", reject)
    cache = {}
    ost, pb, provenance, error = sdf.make_ligand(gemmi.cif.Document(), rows, "CCO", cache, "ensemble")
    assert ost is not None
    assert pb is None
    assert error == "Synthetic sanitization failure"
    assert cache == {}


def test_pb_does_not_run_when_preparation_error_has_a_stale_sdf():
    from unittest.mock import Mock
    from core_score.analysis.posebusters import check_pose

    buster = Mock()
    result = check_pose(buster, [], dict(
        ensemble="e", pose_id="p", path="input.cif",
        prepared=dict(receptor="receptor.pdb", copy_records=[dict(
            copy_id="0", pb_sdf="stale.sdf", pb_error="Identity mismatch", provenance=None)])))
    buster.bust.assert_not_called()
    assert result["pb_valid"] is None
    assert result["copies"][0]["error"] == "Identity mismatch"


def test_peer_graph_recovery_keeps_recipient_coordinates_and_cache_scope(monkeypatch):
    import gemmi
    from core_score.inputs import structures as sdf

    donor = conformer("CCO")
    positions = donor.GetConformer().GetPositions().copy()
    positions[2] += 20  # Distorted pose: distance inference loses the C-O bond.
    rows = [dict(type_symbol=a.GetSymbol(), label_atom_id=f"a{i}",
                 Cartn_x=str(x), Cartn_y=str(y), Cartn_z=str(z))
            for i, (a, (x, y, z)) in enumerate(zip(donor.GetAtoms(), positions))]
    cache = {}
    doc = gemmi.cif.Document()
    monkeypatch.setattr(sdf, "native_graph", lambda *args: (None, None))
    assert sdf.make_ligand(doc, rows, "CCO", cache, "ensemble")[3]

    donor_rows = [dict(row, Cartn_x=str(x), Cartn_y=str(y), Cartn_z=str(z))
                  for row, (x, y, z) in zip(rows, donor.GetConformer().GetPositions())]
    monkeypatch.setattr(sdf, "native_graph", lambda *args: (donor, "native_cif_bonds"))
    assert sdf.make_ligand(doc, donor_rows, "CCO", cache, "ensemble")[3] is None
    monkeypatch.setattr(sdf, "native_graph", lambda *args: (None, None))
    _, molecule, provenance, error = sdf.make_ligand(doc, rows, "CCO", cache, "ensemble")
    assert error is None
    assert provenance == "same_ensemble_atom_name_graph"
    np.testing.assert_array_equal(molecule.GetConformer().GetPositions(), positions)
    # A different ensemble or different declared SMILES cannot borrow this entry.
    assert sdf.make_ligand(doc, rows, "CCO", cache, "other")[3]
    assert sdf.make_ligand(doc, rows, "COC", cache, "ensemble")[3]


def test_identical_chains_get_one_entity_and_atoms_are_untouched():
    from core_score.inputs.structures import atom_rows, merge_identical_entities

    text = """data_pose
loop_
_entity.id
_entity.type
1 polymer
2 polymer
3 polymer
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 GLY
2 1 ALA
2 2 GLY
3 1 ALA
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
C 3
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_entity_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C 1 A 1.234 0 0
ATOM 2 C 2 B 2.345 0 0
ATOM 3 C 3 C 3.456 0 0
"""
    doc = gemmi.cif.read_string(text)
    before = atom_rows(doc)
    merge_identical_entities(doc)
    block = doc.sole_block()
    assert list(block.find_values("_entity.id")) == ["1", "3"]
    assert list(block.find_values("_struct_asym.entity_id")) == ["1", "1", "3"]
    assert [r["label_entity_id"] for r in atom_rows(doc)] == ["1", "1", "3"]
    assert [{k: v for k, v in r.items() if k != "label_entity_id"} for r in atom_rows(doc)] == \
        [{k: v for k, v in r.items() if k != "label_entity_id"} for r in before]


def test_receptor_shortens_chain_names_too_long_for_pdb(tmp_path):
    from pathlib import Path
    from core_score.inputs.structures import write_receptor

    doc = gemmi.cif.read(str(Path(__file__).parent / "fixtures" / "7wl4_pose0.cif"))
    for tag in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"):
        column = doc.sole_block().find_values(tag)
        for i, chain in enumerate(column):
            column[i] = f"1.{chain}"  # PLINDER's chain names
    source = tmp_path / "system.cif"
    doc.write_file(str(source))
    write_receptor(source, set(), tmp_path / "receptor.pdb")
    structure = gemmi.read_structure(str(tmp_path / "receptor.pdb"))
    assert len(structure[0]) == len(gemmi.read_structure(str(source))[0])


ETHANOL = """data_pose
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
HETATM 1 C C1 LIG B 1 0.000 0.000 0.000
HETATM 2 C C2 LIG B 1 1.520 0.000 0.000
HETATM 3 O O1 LIG B 1 2.000 1.350 0.000
"""
AF3_SMILES = """loop_
_chem_comp.id
_chem_comp.pdbx_smiles
LIG CCO
"""
TWO_IDENTITIES = ETHANOL + """HETATM 4 C C1 ETHER C 1 4.000 0.000 0.000
HETATM 5 O O1 ETHER C 1 5.400 0.000 0.000
HETATM 6 C C2 ETHER C 1 6.000 1.200 0.000
loop_
_chem_comp.id
_chem_comp.pdbx_smiles
LIG CCO
ETHER COC
"""


def test_automatic_selection_distinguishes_native_same_composition_ligands():
    from core_score.inputs.structures import choose_ligands, atom_rows

    doc = gemmi.cif.read_string(TWO_IDENTITIES)
    before = atom_rows(doc)
    assert [key[0] for key in choose_ligands(doc, dict(smiles="CCO"))] == ["B"]
    assert [key[0] for key in choose_ligands(doc, dict(smiles="COC"))] == ["C"]
    assert atom_rows(doc) == before


def test_explicit_wrong_chain_still_fails_identity_verification():
    from core_score.inputs.structures import choose_ligands, make_ligand

    doc = gemmi.cif.read_string(TWO_IDENTITIES)
    [rows] = choose_ligands(doc, dict(smiles="CCO", ligand_chains=["C"])).values()
    with pytest.raises(ValueError, match="identity mismatch"):
        make_ligand(doc, rows, "CCO")


def test_auto_selection_retains_candidates_without_verified_native_identity():
    from core_score.inputs.structures import choose_ligands

    # Do not drop a distorted same-composition ligand or infer identity from
    # coordinate connectivity: both copies must reach chemistry verification.
    doc = gemmi.cif.read_string(TWO_IDENTITIES.replace("ETHER COC", "ETHER ?"))
    assert len(choose_ligands(doc, dict(smiles="CCO"))) == 2


def test_auto_selection_keeps_all_identical_native_copies():
    from core_score.inputs.structures import choose_ligands

    doc = gemmi.cif.read_string(TWO_IDENTITIES.replace("ETHER COC", "ETHER CCO"))
    assert len(choose_ligands(doc, dict(smiles="CCO"))) == 2


BOND_TABLE = """loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
LIG C1 C2 SING
LIG C2 O1 SING
"""


@pytest.mark.parametrize("extra, route", [
    (AF3_SMILES, "native_smiles_atom_names"),  # AlphaFold3
    (BOND_TABLE, "native_cif_bonds"),  # OpenFold3
    ("", "coordinate_connectivity"),  # Boltz, Chai-1, Protenix
])
def test_each_bond_route_gives_the_same_chemistry_at_the_predicted_coordinates(extra, route):
    from core_score.inputs.structures import choose_ligands, make_ligand

    doc = gemmi.cif.read_string(ETHANOL + extra)
    [rows] = choose_ligands(doc, dict(smiles="CCO")).values()
    _, pb_mol, provenance, error = make_ligand(doc, rows, "CCO", {}, "ensemble")
    assert (provenance, error) == (route, None)
    assert canonical_smiles(pb_mol) == "CCO"
    assert pb_mol.GetConformer().GetPositions().tolist() == [[0, 0, 0], [1.52, 0, 0], [2, 1.35, 0]]


def test_pdb_input_is_prepared_like_cif(tmp_path):
    import json
    from pathlib import Path
    from core_score.inputs.prepare import prepare_pose

    fixtures = Path(__file__).parent / "fixtures"
    smiles = json.loads((fixtures / "manifest.json").read_text())["smiles"]
    pdb = tmp_path / "pose.pdb"
    gemmi.read_structure(str(fixtures / "7wl4_pose0.cif")).write_pdb(str(pdb))
    pose = dict(ensemble="e", pose_id="p", smiles=smiles, ligand_chains=["B"])
    from_cif = prepare_pose(dict(pose, path=str(fixtures / "7wl4_pose0.cif")), tmp_path / "cif" / "pose.cif")
    from_pdb = prepare_pose(dict(pose, path=str(pdb)), tmp_path / "pdb" / "pose.cif")
    assert from_pdb["copies"] == from_cif["copies"] == 1
    assert from_pdb["copy_records"][0]["pb_error"] is None
    assert from_pdb["copy_records"][0]["centre"] == pytest.approx(from_cif["copy_records"][0]["centre"], abs=1e-3)


def test_ost_uses_verified_bonds_even_when_coordinate_inference_loses_a_bond(tmp_path):
    from core_score.inputs.prepare import prepare
    from core_score.inputs.sites import site_pose

    source = tmp_path / "pose.cif"
    source.write_text((ETHANOL + BOND_TABLE).replace("2.000 1.350", "20.000 1.350"))
    original = source.read_bytes()
    result = prepare(dict(path=str(source), smiles="CCO", ensemble="e", pose_id="p"), tmp_path / "out.cif")
    assert not result.get("error")
    prepared = result["prepared"]
    copy = prepared["copy_records"][0]
    assert prepared["ligand_sdfs"] == [copy["pb_sdf"]]
    assert Chem.MolFromMolFile(copy["pb_sdf"]).GetNumBonds() == 2
    assert Chem.MolFromMolFile(copy["ost_sdf"], sanitize=False).GetNumBonds() == 1
    assert copy["coordinate_audit"]["max_coordinate_difference_angstrom"] <= 5.00001e-5
    split = site_pose(result, copy, tmp_path / "site.cif")
    assert split["prepared"]["ligand_sdfs"] == [copy["pb_sdf"]]
    assert source.read_bytes() == original


def test_unverified_graph_is_not_silently_scored_with_coordinate_bonds(tmp_path):
    from core_score.inputs.prepare import prepare

    source = tmp_path / "pose.cif"
    source.write_text(ETHANOL.replace("2.000 1.350", "20.000 1.350"))
    result = prepare(dict(path=str(source), smiles="CCO", ensemble="e", pose_id="p"), tmp_path / "out.cif")
    assert "Cannot verify ligand connectivity" in result["error"]["error"]


def test_verified_graph_from_later_pose_recovers_earlier_pose(tmp_path):
    from core_score.inputs.prepare import prepare_groups

    first, second = tmp_path / "a.cif", tmp_path / "b.cif"
    first.write_text(ETHANOL.replace("2.000 1.350", "20.000 1.350"))
    second.write_text(ETHANOL + BOND_TABLE)
    poses = [dict(path=str(p), smiles="CCO", ensemble="e", pose_id=p.name) for p in (first, second)]
    [group] = prepare_groups([dict(ensemble="e", poses=poses)], tmp_path / "out")
    assert not any(p.get("error") for p in group["poses"])
    recovered = group["poses"][0]["prepared"]["copy_records"][0]
    assert recovered["provenance"] == "same_ensemble_atom_name_graph"
    mol = Chem.MolFromMolFile(recovered["pb_sdf"])
    assert mol.GetConformer().GetAtomPosition(2).x == 20
