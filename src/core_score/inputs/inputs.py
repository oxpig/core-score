"""Load ensembles from CSV or structure paths; resolve paths relative to the CSV."""

import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from core_score.analysis.core import NEIGHBOURS
from rdkit import Chem
from core_score.inputs.structures import template_mol

REQUIRED = ("ensemble", "ligand", "predictions")
OPTIONAL = ("confidence", "ligand_chain", "reference")
MIN_POSES = NEIGHBOURS + 1
LIGAND_FILES = (".sdf", ".mol")


def load_inputs(inputs, *, ligand=None, ligand_chains=None, reference=None):
    """Load poses in numeric path order, fixing site anchors and OST pair direction."""
    inputs = [Path(path) for path in inputs]
    if len(inputs) == 1 and inputs[0].suffix.lower() == ".csv":
        base, rows = inputs[0].resolve().parent, read_csv(inputs[0])
    else:
        base = Path.cwd()
        rows = [dict(ensemble="ensemble", ligand=ligand or "", predictions=str(path),
                     ligand_chain=" ".join(ligand_chains or []), reference=str(reference or ""))
                for path in inputs]
    ensembles = defaultdict(list)
    for row in rows:
        ensembles[row["ensemble"]].append({k: v.strip() for k, v in row.items() if v and v.strip()})
    groups = [make_group(name, members, base) for name, members in ensembles.items()]
    small = [f"{group['ensemble']} ({len(group['poses'])})" for group in groups if len(group["poses"]) < MIN_POSES]
    if small:
        raise ValueError(f"CORE needs at least {MIN_POSES} predictions per ensemble, since each is scored "
                         f"against its {NEIGHBOURS} closest others. Too few: {', '.join(small)}")
    return groups


def read_csv(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [c for c in REQUIRED if c not in columns]
        unknown = [c for c in columns if c not in REQUIRED + OPTIONAL]
        if missing or unknown:
            raise ValueError(f"{path}: missing columns {missing}, unknown columns {unknown}. "
                             f"The CSV needs {', '.join(REQUIRED)} and may add {', '.join(OPTIONAL)}.")
        rows = list(reader)
    blank = [i + 2 for i, row in enumerate(rows) if not all((row[c] or "").strip() for c in REQUIRED)]
    if blank:
        raise ValueError(f"{path}: lines {blank} leave ensemble, ligand or predictions empty")
    return rows


def make_group(name, rows, base):
    """One ensemble from its CSV rows. Every row must give the same ligand and reference."""
    for column in ("ligand", "reference"):
        if len({row.get(column) for row in rows}) > 1:
            raise ValueError(f"{name}: every row of an ensemble needs the same {column}")
    try:
        smiles = ligand_smiles(rows[0].get("ligand", ""), base)
        template_mol(smiles)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{name}: {exc}. Give the ligand as a SMILES or an SDF file in the CSV, "
                         "or --ligand with structure files.") from exc
    poses = [pose for row in rows for pose in make_poses(name, row, base, smiles)]
    paths = [pose["path"] for pose in poses]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name}: a structure is listed more than once")
    if len({"confidence" in pose for pose in poses}) != 1:
        raise ValueError(f"{name}: give a confidence for every prediction or for none")
    group = dict(ensemble=name, poses=sorted(poses, key=lambda pose: natural_path_key(pose["path"])))
    if rows[0].get("reference"):
        path = resolve(base, rows[0]["reference"])
        group["reference"] = dict(ensemble=name, pose_id="reference", path=path, smiles=smiles)
    return group


def natural_path_key(path):
    """Numeric seed/sample order; break leading-zero ties by the original path."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)), path


def ligand_smiles(ligand, base):
    """The ligand's SMILES, given either as a SMILES or as the path to an SDF or MOL file."""
    if Path(ligand).suffix.lower() not in LIGAND_FILES:
        return ligand
    molecules = list(Chem.SDMolSupplier(resolve(base, ligand)))
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"{ligand} must hold one readable molecule")
    return Chem.MolToSmiles(molecules[0])


def make_poses(name, row, base, smiles):
    """The predictions one CSV row names: one file, or every file its wildcard matches."""
    paths = sorted(glob.glob(str(base / row["predictions"]), recursive=True))
    if not paths:
        raise FileNotFoundError(f"{name}: no structure file matches {row['predictions']}")
    if "confidence" in row and len(paths) > 1:
        raise ValueError(f"{name}: {row['predictions']} matches {len(paths)} files, so give "
                         "confidence on a row of its own for each prediction")
    poses = []
    for path in paths:
        path = str(Path(path).resolve())
        pose = dict(ensemble=name, pose_id=Path(os.path.relpath(path, base)).as_posix(), path=path,
                    smiles=smiles, ligand_chains=row.get("ligand_chain", "").split())
        if "confidence" in row:
            pose["confidence"] = float(row["confidence"])
        poses.append(pose)
    return poses


def resolve(base, path):
    return str((base / path).resolve(strict=True))
