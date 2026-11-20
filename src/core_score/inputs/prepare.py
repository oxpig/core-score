"""Prepare every pose: select its ligand copies, give identical chains one entity, write
verified SDFs for OST and PoseBusters and a receptor for PoseBusters.
Predicted coordinates are never changed apart from SDF serialization precision."""

from pathlib import Path
import numpy as np
from core_score.outputs.files import diagnostic, pose_directory
from core_score.inputs.structures import (
    choose_ligands, filter_records, heavy_atoms, make_ligand, merge_identical_entities, read_document,
    write_molecule, write_receptor,
)


def prepare_groups(groups, output, *, timeout=None):
    cache = {}
    prepared = []
    for group in groups:
        poses = [prepare(pose, pose_directory(output, pose) / "pose.cif", cache, timeout) for pose in group["poses"]]
        # A ligand graph that failed can borrow one from a peer prepared later in the same ensemble.
        for i, pose in enumerate(poses):
            if "prepared" in pose and any(copy["pb_error"] for copy in pose["prepared"]["copy_records"]):
                retry = prepare(group["poses"][i], pose_directory(output, pose) / "pose.cif", cache, timeout)
                if "prepared" in retry:
                    poses[i] = retry
        prepared.append(dict(group, poses=poses))
    return prepared


def prepare(pose, destination, cache=None, timeout=None):
    """The pose with its prepared record, or with the error that stopped preparation."""
    prepared = None
    try:
        prepared = prepare_pose(pose, destination, cache)
        failures = [copy["pb_error"] or "Missing verified ligand SDF"
                    for copy in prepared["copy_records"] if copy["pb_error"] or not copy["pb_sdf"]]
        if failures:
            raise ValueError("Cannot verify ligand connectivity: " + "; ".join(failures))
        prepared.update(ligand_sdfs=[copy["pb_sdf"] for copy in prepared["copy_records"]],
                        route="smiles_matched_explicit_connectivity_v1")
        return dict(pose, prepared=prepared)
    except Exception as exc:
        result = dict(pose, error=diagnostic("preparation", exc, structure=pose["path"]))
        if prepared is not None:
            result["prepared"] = prepared  # A later peer may supply a verified graph.
        return result


def prepare_pose(pose, destination, cache=None):
    """Prepared record for one pose; its files are written beside `destination`."""
    source = Path(pose["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = read_document(source)
    merge_identical_entities(doc)
    prepared, ligands = from_structure(pose, doc, destination, cache)
    receptor = destination.with_suffix(".receptor.pdb")
    write_receptor(source, ligands, receptor)
    return dict(prepared, receptor=str(receptor))


def from_structure(pose, doc, destination, cache):
    """Ligand residues selected in the structure, with chemistry built from their atoms."""
    selected = choose_ligands(doc, pose)
    copies = []
    for i, (key, rows) in enumerate(selected.items()):
        heavy = heavy_atoms(rows)
        ost_mol, pb_mol, provenance, pb_error = make_ligand(doc, rows, pose["smiles"], cache, pose["ensemble"])
        ost_sdf = destination.with_name(f"{destination.stem}.copy{i}.ost.sdf")
        write_molecule(ost_mol, ost_sdf)
        pb_sdf, coordinate_audit = None, None
        if pb_mol is not None:
            pb_sdf = destination.with_name(f"{destination.stem}.copy{i}.pb.sdf")
            coordinate_audit = write_molecule(pb_mol, pb_sdf)
            pb_sdf = str(pb_sdf)
        centre = np.mean([[float(r["Cartn_" + c]) for c in "xyz"] for r in heavy], axis=0)
        copies.append(dict(copy_id=str(i), residue=list(key), ost_sdf=str(ost_sdf), pb_sdf=pb_sdf,
                           pb_error=pb_error, provenance=provenance, centre=centre.tolist(),
                           coordinate_audit=coordinate_audit))
    filter_records(doc, set(selected), destination)
    prepared = dict(path=str(destination), ligand_sdfs=[], copies=len(copies), copy_records=copies, route="native_cif")
    return prepared, set(selected)
