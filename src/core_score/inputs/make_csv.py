"""Build input CSVs from model outputs, deduplicating structures and reading ligand ipTM."""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
from core_score.inputs.inputs import LIGAND_FILES, MIN_POSES, ligand_smiles, resolve
from core_score.inputs.structures import atom_rows, choose_ligands, ligand_residues, read_document, residue_key

STRUCTURE_SUFFIXES = (".cif", ".pdb")


def read_json(path):
    return json.loads(Path(path).read_text())


def as_matrix(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)


def af3_or_protenix(path):
    return as_matrix(read_json(path)["chain_pair_iptm"])


def boltz(path):
    pairs = read_json(path)["pair_chains_iptm"]
    return as_matrix([[pairs[i][j] for j in sorted(pairs[i], key=int)] for i in sorted(pairs, key=int)])


def chai(path):
    return np.load(path)["per_chain_pair_iptm"][0].astype(float)


def openfold3(path):
    """OpenFold3 names each pair of chains, (A, B), rather than giving a matrix."""
    return read_json(path)["chain_pair_iptm"]


# (model, structure file name, its confidence file name, reader), tried in order.
CONFIDENCE_FILES = [
    ("AlphaFold3", r"(.*)model\.cif", r"\1summary_confidences.json", af3_or_protenix),
    ("OpenFold3", r"(.*)model\.cif", r"\1confidences_aggregated.json", openfold3),
    ("Boltz", r"(.+)_model_(\d+)\.(cif|pdb)", r"confidence_\1_model_\2.json", boltz),
    ("Chai-1", r"pred\.model_idx_(\d+)\.cif", r"scores.model_idx_\1.npz", chai),
    ("Protenix", r"(.+)_sample_(\d+)\.cif", r"\1_summary_confidence_sample_\2.json", af3_or_protenix),
]


def confidence_file(structure):
    """The model and confidence file written beside a structure, or (None, None, None)."""
    for model, pattern, name, reader in CONFIDENCE_FILES:
        match = re.fullmatch(pattern, structure.name)
        if match and (structure.parent / match.expand(name)).is_file():
            return model, structure.parent / match.expand(name), reader
    return None, None, None


def ligand_iptm(structure, smiles, rows, scores):
    """Mean chain-pair ipTM between the ligand's chains and the receptor chains.

    Chains are matched to the confidence matrix in the order they appear in the structure.
    Receptor chains are those holding anything other than free ligands.
    """
    doc = read_document(structure)
    chains = list(dict.fromkeys(residue_key(row)[0] for row in rows))
    ligands = sorted({key[0] for key in choose_ligands(doc, dict(smiles=smiles))})
    loose = set(ligand_residues(doc))
    receptor = list(dict.fromkeys(residue_key(row)[0] for row in rows if residue_key(row) not in loose))
    if isinstance(scores, dict):
        def pair(a, b):
            return scores.get(f"({a}, {b})", scores.get(f"({b}, {a})", np.nan))
    else:
        if scores.shape != (len(chains), len(chains)):
            raise ValueError(f"the confidence file scores {len(scores)} chains but {structure.name} has {len(chains)}")
        def pair(a, b):
            return scores[chains.index(a), chains.index(b)]
    values = [pair(a, b) for a in ligands for b in receptor]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError(f"no ligand to receptor ipTM for ligand chains {ligands} and receptor chains {receptor}")
    return float(np.mean(values))


def find_poses(folder):
    """Structure files under `folder`, sorted, each set of coordinates once. Of identical
    files, the one deepest in the folder tree is kept, since it sits beside its own sample's
    files. Returns the poses and the duplicates left out."""
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in STRUCTURE_SUFFIXES)
    kept, skipped = {}, []
    for path in sorted(files, key=lambda p: (-len(p.parts), str(p))):
        rows = atom_rows(read_document(path))
        coordinates = ",".join(f"{float(r['Cartn_' + c]):.3f}" for r in rows for c in "xyz")
        key = hashlib.sha256(coordinates.encode()).hexdigest()
        if key in kept:
            skipped.append(path)
        else:
            kept[key] = (path, rows)
    return sorted(kept.values()), skipped


def read_ligands(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"ensemble", "ligand"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path} needs the columns ensemble,ligand")
        return {row["ensemble"].strip(): row["ligand"].strip() for row in reader if row["ligand"].strip()}


def build_rows(predictions, ligands_csv, output, references=None):
    """One CSV row per prediction, and notes to print about what was found."""
    predictions, output = Path(predictions), Path(output)
    ligands, base = read_ligands(ligands_csv), Path(ligands_csv).resolve().parent
    folders = sorted(p for p in predictions.iterdir() if p.is_dir())
    if not folders:
        raise ValueError(f"{predictions} has no folders; put each ensemble's structures in a folder named for it")
    problems = [f"{f.name}: no ligand in {ligands_csv}" for f in folders if f.name not in ligands]
    smiles, written = {}, {}
    for name, ligand in ligands.items():
        try:
            smiles[name] = ligand_smiles(ligand, base)
            in_file = Path(ligand).suffix.lower() in LIGAND_FILES
            written[name] = relative(resolve(base, ligand), output) if in_file else ligand
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: {exc}")
    rows, notes, models = [], [], set()
    for folder in folders:
        poses, skipped = find_poses(folder)
        notes += [f"{folder.name}: left out {p.relative_to(predictions)}, a copy of another structure" for p in skipped]
        if len(poses) < MIN_POSES:
            problems.append(f"{folder.name}: {len(poses)} structures, CORE needs at least {MIN_POSES}")
        reference = None
        if references:
            found = [Path(references) / f"{folder.name}{s}" for s in STRUCTURE_SUFFIXES]
            found = [p for p in found if p.is_file()]
            if not found:
                problems.append(f"{folder.name}: no {folder.name}.cif or {folder.name}.pdb in {references}")
            reference = found[0] if found else None
        for path, atoms in poses:
            row = dict(ensemble=folder.name, ligand=written.get(folder.name, ""), predictions=relative(path, output))
            model, scores_path, reader = confidence_file(path)
            if model and folder.name in smiles:
                try:
                    row["confidence"] = ligand_iptm(path, smiles[folder.name], atoms, reader(scores_path))
                except (ValueError, KeyError) as exc:
                    problems.append(f"{path.relative_to(predictions)}: confidence not read from {scores_path.name}: {exc}")
                models.add(model)
            if reference:
                row["reference"] = relative(reference, output)
            rows.append(row)
    with_confidence = sum("confidence" in row for row in rows)
    if 0 < with_confidence < len(rows):
        missing = [row["predictions"] for row in rows if "confidence" not in row]
        problems.append(f"confidence found for {with_confidence} of {len(rows)} predictions; none beside {missing[:5]}")
    if problems:
        raise ValueError("cannot write the CSV:\n  " + "\n  ".join(problems))
    notes.append(f"confidence: ligand ipTM read from {', '.join(sorted(models))} files" if models else
                 "confidence: no confidence files recognised, so the CSV has no confidence column")
    return rows, notes


def relative(path, output):
    return Path(os.path.relpath(Path(path).resolve(), output.resolve().parent)).as_posix()


def parser():
    p = argparse.ArgumentParser(
        prog="core-score-csv",
        description="Write the CSV core-score reads from a folder holding one folder of predictions per ensemble.",
        epilog="Example: core-score-csv --predictions predictions/ --ligands ligands.csv --output core_score_input.csv",
    )
    p.add_argument("--predictions", type=Path, required=True, help="Folder with one folder per ensemble")
    p.add_argument("--ligands", type=Path, required=True,
                   help="CSV with the columns ensemble,ligand, each ligand a SMILES or an SDF file")
    p.add_argument("--output", type=Path, required=True, help="CSV to write")
    p.add_argument("--references", type=Path, help="Folder holding <ensemble>.cif or <ensemble>.pdb ground truths")
    return p


def main(argv=None):
    words = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(words or ["--help"])
    if args.output.exists():
        print(f"core-score-csv: {args.output} already exists; choose a new file", file=sys.stderr)
        return 2
    try:
        rows, notes = build_rows(args.predictions, args.ligands, args.output, args.references)
    except (ValueError, OSError) as exc:
        print(f"core-score-csv: {exc}", file=sys.stderr)
        return 2
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    for note in notes:
        print(note)
    ensembles = len({row["ensemble"] for row in rows})
    print(f"Wrote {len(rows)} predictions in {ensembles} ensemble{'' if ensembles == 1 else 's'} to {args.output}")
    print(f"Next: core-score {args.output} --out results/")
    return 0
