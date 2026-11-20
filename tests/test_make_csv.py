import csv
import json
import pytest
from core_score.inputs.make_csv import build_rows, main

CIF = """data_pose
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
ATOM 1 C CA ALA A 1 0 0 0
HETATM 2 C C1 LIG B 1 {x} 0 0
"""


def af3_ensemble(folder, iptms, top_copy=True):
    """AlphaFold3's layout: a folder per sample, and a copy of the first sample's files on top."""
    for sample, iptm in enumerate(iptms):
        directory = folder / f"seed-1_sample-{sample}"
        directory.mkdir(parents=True)
        (directory / "model.cif").write_text(CIF.format(x=sample + 1))
        (directory / "summary_confidences.json").write_text(json.dumps({"chain_pair_iptm": [[0.9, 0.1], [iptm, 0.5]]}))
    if top_copy:
        (folder / "job_model.cif").write_text(CIF.format(x=1))
        (folder / "job_summary_confidences.json").write_text((folder / "seed-1_sample-0/summary_confidences.json").read_text())


def ligands_csv(path, names):
    path.write_text("ensemble,ligand\n" + "".join(f"{name},C\n" for name in names))
    return path


def test_af3_folders_give_one_row_per_sample_with_ligand_iptm(tmp_path):
    af3_ensemble(tmp_path / "preds" / "target", [0.8, 0.6, 0.7, 0.2])
    rows, notes = build_rows(tmp_path / "preds", ligands_csv(tmp_path / "s.csv", ["target"]), tmp_path / "in.csv")
    assert [r["predictions"] for r in rows] == [f"preds/target/seed-1_sample-{i}/model.cif" for i in range(4)]
    assert [r["confidence"] for r in rows] == [0.8, 0.6, 0.7, 0.2]  # ligand row, receptor column
    assert any("left out target/job_model.cif" in note for note in notes)


def test_openfold3_names_chain_pairs(tmp_path):
    folder = tmp_path / "preds" / "target" / "seed_1"
    folder.mkdir(parents=True)
    for sample in range(4):
        (folder / f"t_seed_1_sample_{sample}_model.cif").write_text(CIF.format(x=sample))
        (folder / f"t_seed_1_sample_{sample}_confidences_aggregated.json").write_text(
            json.dumps({"chain_pair_iptm": {"(A, B)": sample / 10}}))
    rows, notes = build_rows(tmp_path / "preds", ligands_csv(tmp_path / "s.csv", ["target"]), tmp_path / "in.csv")
    assert [r["confidence"] for r in rows] == [0.0, 0.1, 0.2, 0.3]
    assert "OpenFold3" in notes[-1]


def test_problems_are_listed_together(tmp_path):
    af3_ensemble(tmp_path / "preds" / "small", [0.8, 0.6, 0.7], top_copy=False)
    af3_ensemble(tmp_path / "preds" / "nameless", [0.8, 0.6, 0.7, 0.2])
    (tmp_path / "refs").mkdir()
    with pytest.raises(ValueError) as error:
        build_rows(tmp_path / "preds", ligands_csv(tmp_path / "s.csv", ["small"]), tmp_path / "in.csv", tmp_path / "refs")
    message = str(error.value)
    assert "nameless: no ligand" in message
    assert "small: 3 structures, CORE needs at least 4" in message
    assert "small: no small.cif or small.pdb" in message


def test_structures_without_confidence_files_leave_the_column_out_and_core_score_reads_the_csv(tmp_path):
    folder = tmp_path / "preds" / "target"
    folder.mkdir(parents=True)
    for i in range(4):
        (folder / f"pose{i}.cif").write_text(CIF.format(x=i))
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "target.cif").write_text(CIF.format(x=0))
    output = tmp_path / "out" / "inputs.csv"
    output.parent.mkdir()
    assert main(["--predictions", str(tmp_path / "preds"), "--ligands", str(ligands_csv(tmp_path / "s.csv", ["target"])),
                 "--output", str(output), "--references", str(tmp_path / "refs")]) == 0
    rows = list(csv.DictReader(output.open()))
    assert list(rows[0]) == ["ensemble", "ligand", "predictions", "reference"]
    assert rows[0]["predictions"] == "../preds/target/pose0.cif" and rows[0]["reference"] == "../refs/target.cif"

    from core_score.inputs.inputs import load_inputs
    [group] = load_inputs([output])
    assert len(group["poses"]) == 4 and group["reference"]["path"] == str(tmp_path / "refs" / "target.cif")


def test_boltz_and_chai_confidence_files_are_read(tmp_path):
    import numpy as np

    boltz = tmp_path / "preds" / "boltz_target" / "predictions" / "job"
    chai = tmp_path / "preds" / "chai_target"
    boltz.mkdir(parents=True)
    chai.mkdir(parents=True)
    for k in range(4):
        (boltz / f"job_model_{k}.cif").write_text(CIF.format(x=k))
        pairs = {"0": {"0": 0.9, "1": 0.2}, "1": {"0": k / 10, "1": 0.5}}
        (boltz / f"confidence_job_model_{k}.json").write_text(json.dumps({"pair_chains_iptm": pairs}))
        (chai / f"pred.model_idx_{k}.cif").write_text(CIF.format(x=k))
        np.savez(chai / f"scores.model_idx_{k}.npz", per_chain_pair_iptm=np.array([[[0.9, 0.2], [k / 20, 0.5]]]))
    rows, notes = build_rows(tmp_path / "preds", ligands_csv(tmp_path / "s.csv", ["boltz_target", "chai_target"]),
                             tmp_path / "in.csv")
    confidence = {}
    for row in rows:
        confidence.setdefault(row["ensemble"], []).append(row["confidence"])
    assert confidence == {"boltz_target": [0.0, 0.1, 0.2, 0.3], "chai_target": [0.0, 0.05, 0.1, 0.15]}
    assert "Boltz" in notes[-1] and "Chai-1" in notes[-1]
