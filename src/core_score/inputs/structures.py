"""Reading predicted structures and building ligands: CIF records, ligand selection and
chemistry, and the SDF files written for OST and PoseBusters. Coordinates are never changed."""

from collections import Counter
from collections import defaultdict
from pathlib import Path
import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDetermineBonds


def read_document(path):
    path = Path(path)
    if path.suffix.lower() in (".pdb", ".ent"):
        structure = gemmi.read_structure(str(path))
        if len(structure) != 1:
            raise ValueError("Supply one structural model per pose file")
        structure.setup_entities()
        return structure.make_mmcif_document()
    doc = gemmi.cif.read_file(str(path))
    doc.sole_block()
    return doc


def merge_identical_entities(doc):
    """Give polymer chains with the same sequence one entity, changing only entity IDs.

    OST treats chains of different entities as different molecules when it maps one
    structure's chains onto another's. AlphaFold3 writes an entity per chain listed in its
    input, so without this the two chains of a homodimer could never swap places.
    """
    block = doc.sole_block()
    sequences = defaultdict(list)
    for entity, number, monomer in block.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"]):
        sequences[entity].append((int(number), monomer))
    first, rename = {}, {}
    for entity, residues in sequences.items():
        rename[entity] = first.setdefault(tuple(sorted(residues)), entity)
    duplicates = {entity for entity, kept in rename.items() if entity != kept}
    if not duplicates:
        return
    for tag in ("_atom_site.label_entity_id", "_struct_asym.entity_id", "_pdbx_poly_seq_scheme.entity_id"):
        column = block.find_values(tag)
        for i, value in enumerate(column):
            if value in duplicates:
                column[i] = rename[value]
    for category, tag in (("_entity.", "id"), ("_entity_poly.", "entity_id"), ("_entity_poly_seq.", "entity_id")):
        table = block.find(category, [tag])
        for i in range(len(table) - 1, -1, -1):
            if table[i][0] in duplicates:
                table.remove_row(i)


def atom_rows(doc):
    table = doc.sole_block().find_mmcif_category("_atom_site.")
    if not len(table):
        raise ValueError("No atom records in structure")
    tags = [str(t).split(".", 1)[1] for t in table.tags]
    rows = [dict(zip(tags, (gemmi.cif.as_string(v) for v in row))) for row in table]
    models = {r.get("pdbx_PDB_model_num", "1") for r in rows}
    if len(models) != 1:
        raise ValueError("Supply one structural model per pose file")
    return rows


def residue_key(row):
    return (
        row.get("auth_asym_id") or row.get("label_asym_id", ""),
        row.get("auth_seq_id") or row.get("label_seq_id", ""),
        row.get("pdbx_PDB_ins_code", ""),
        row.get("label_comp_id") or row.get("auth_comp_id", ""),
    )


def ligand_residues(doc):
    block = doc.sole_block()
    polymer = {
        gemmi.cif.as_string(v) for v in block.find_values("_entity_poly.entity_id")
    }
    groups = defaultdict(list)
    for row in atom_rows(doc):
        if row["group_PDB"] == "HETATM" and row.get("label_entity_id") not in polymer:
            groups[residue_key(row)].append(row)
    return dict(groups)


def filter_records(doc, selected, destination):
    """Keep all polymer records and selected ligand residues, byte-value exact in CIF."""
    rows = atom_rows(doc)
    table = doc.sole_block().find_mmcif_category("_atom_site.")
    tags = list(table.tags)
    nonpoly = set(ligand_residues(doc))
    keep = [
        r["group_PDB"] != "HETATM"
        or residue_key(r) not in nonpoly
        or residue_key(r) in selected
        for r in rows
    ]
    expected = [list(row) for row, retained in zip(table, keep) if retained]
    for i in range(len(rows) - 1, -1, -1):
        if not keep[i]:
            table.remove_row(i)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.write_file(str(destination))
    actual = (
        gemmi.cif.read_file(str(destination))
        .sole_block()
        .find_mmcif_category("_atom_site.")
    )
    if list(actual.tags) != tags or [list(row) for row in actual] != expected:
        raise ValueError("CIF serialization changed a retained atom record")


def write_receptor(source, ligands, destination):
    """The predicted environment without the selected ligand residues, as PDB for PoseBusters.
    Chain IDs too long for PDB, such as PLINDER's 1.A, are shortened."""
    doc = read_document(source)
    table = doc.sole_block().find_mmcif_category("_atom_site.")
    for i, row in reversed(list(enumerate(atom_rows(doc)))):
        if row["group_PDB"] == "HETATM" and residue_key(row) in ligands:
            table.remove_row(i)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    if any(len(chain.name) > 2 for model in structure for chain in model):  # gemmi crashes on a structure with no models
        structure.shorten_chain_names()
    structure.write_pdb(str(destination))


def template_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumHeavyAtoms():
        raise ValueError("ligand SMILES is missing or invalid")
    return Chem.RemoveAllHs(mol)


def heavy_atoms(rows):
    return [r for r in rows if r["type_symbol"].capitalize() not in ("H", "D")]


def atom_name(row):
    return row.get("label_atom_id", row.get("auth_atom_id"))


def composition(rows):
    """Heavy-atom element counts of a residue's atom records."""
    return Counter(r["type_symbol"].capitalize() for r in heavy_atoms(rows))


def choose_ligands(doc, pose):
    """Ligand residues picked out by the pose's chains and its SMILES composition."""
    candidates = ligand_residues(doc)
    selected = candidates
    if pose.get("ligand_chains"):
        missing = set(pose["ligand_chains"]) - {key[0] for key in candidates}
        if missing:
            raise ValueError(f"Ligand chains absent: {sorted(missing)}; candidates: {list(candidates)}")
        selected = {k: v for k, v in selected.items() if k[0] in pose["ligand_chains"]}
    template = template_mol(pose["smiles"])
    expected = Counter(a.GetSymbol() for a in template.GetAtoms())
    selected = {k: v for k, v in selected.items() if composition(v) == expected}
    if not pose.get("ligand_chains") and len(selected) > 1:
        # Reject isomers only from native chemistry, never from distorted geometry.
        identity = canonical_smiles(template)
        native_matches, native_mismatches = set(), set()
        for key, rows in list(selected.items()):
            graph, route = native_graph(doc, rows)
            if route == "native_smiles_atom_names" and canonical_smiles(graph) != identity:
                del selected[key]
            elif route == "native_cif_bonds":
                try:
                    matched_template(template, graph)
                except ValueError:
                    native_mismatches.add(key)
                else:
                    native_matches.add(key)
        # Bond-table exclusions require at least one verified matching residue.
        if native_matches:
            selected = {k: v for k, v in selected.items() if k not in native_mismatches}
    if not selected:
        raise ValueError(f"Selected ligand absent; candidates: {list(candidates)}")
    if len({tuple(sorted(composition(rows).items())) for rows in selected.values()}) > 1:
        raise ValueError(
            "Selected residues contain different ligand compositions; give each ligand identity its own ensemble"
        )
    for key, rows in selected.items():
        names = [atom_name(r) for r in heavy_atoms(rows)]
        if len(names) != len(set(names)):
            raise ValueError(f"Alternate conformers or duplicate ligand atom names in {key}; supply one conformer")
    return selected


def coordinate_mol(rows):
    """The heavy atoms as an RDKit molecule with their exact coordinates and no bonds."""
    rows = heavy_atoms(rows)
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(rows))
    for i, row in enumerate(rows):
        atom = Chem.Atom(row["type_symbol"].capitalize())
        atom.SetNoImplicit(True)
        atom.SetProp(
            "_CIF_ATOM_NAME", row.get("label_atom_id", row.get("auth_atom_id", str(i)))
        )
        rw.AddAtom(atom)
        xyz = [float(row["Cartn_" + c]) for c in "xyz"]
        if not np.isfinite(xyz).all():
            raise ValueError("Non-finite ligand coordinates")
        conf.SetAtomPosition(i, Point3D(*xyz))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    return mol


def canonical_smiles(mol):
    """Canonical SMILES without hydrogens or stereochemistry, for comparing chemical identity."""
    mol = Chem.RemoveAllHs(Chem.Mol(mol))
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def native_graph(doc, rows):
    """Use CIF bond tables or AF3's native SMILES/name bijection when available."""
    names = [atom_name(r) for r in heavy_atoms(rows)]
    comp = rows[0].get("label_comp_id", rows[0].get("auth_comp_id"))
    block = doc.sole_block()
    table = block.find_mmcif_category("_chem_comp_bond.")
    if len(table):
        tags = [str(t).split(".", 1)[1] for t in table.tags]
        bonds = [dict(zip(tags, (str(v).strip("\"'") for v in row))) for row in table]
        bonds = [b for b in bonds if b.get("comp_id") == comp]
        if bonds:
            mol = Chem.RWMol(coordinate_mol(rows))
            orders = {
                "SING": Chem.BondType.SINGLE,
                "DOUB": Chem.BondType.DOUBLE,
                "TRIP": Chem.BondType.TRIPLE,
                "AROM": Chem.BondType.AROMATIC,
            }
            for b in bonds:
                a, c = b["atom_id_1"], b["atom_id_2"]
                if a in names and c in names:
                    order = b.get("value_order", "").upper()
                    if order not in orders:
                        raise ValueError(f"Unsupported native bond order {order}")
                    mol.AddBond(names.index(a), names.index(c), orders[order])
            return mol.GetMol(), "native_cif_bonds"
    table = block.find_mmcif_category("_chem_comp.")
    tags = [str(t).split(".", 1)[1] for t in table.tags]
    for record in table:
        row = dict(zip(tags, (str(v).strip("\"'") for v in record)))
        if row.get("id") == comp and row.get("pdbx_smiles") not in (None, "?", "."):
            mol = template_mol(row["pdbx_smiles"])
            counts, expected = Counter(), []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                counts[symbol] += 1
                expected.append(symbol + str(counts[symbol]))
            if len(names) == len(expected) and set(names) == set(expected):
                mol = Chem.RenumberAtoms(mol, [expected.index(n) for n in names])
                mol.AddConformer(Chem.Conformer(coordinate_mol(rows).GetConformer()))
                return mol, "native_smiles_atom_names"
    return None, None


def generic_graph(mol):
    result = Chem.RWMol(Chem.Mol(mol))
    Chem.RemoveStereochemistry(result)
    for atom in result.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(True)
    for bond in result.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)
    mol = result.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def matched_template(template, observed):
    """Transfer the exact template graph only after an element-labelled graph bijection.

    Retains dative-bond direction, charges, radical state and hydrogen counts;
    never changes coordinates or adds a missing observed bond.
    """
    left, right = generic_graph(template), generic_graph(observed)
    if (left.GetNumAtoms(), left.GetNumBonds()) != (
        right.GetNumAtoms(),
        right.GetNumBonds(),
    ):
        raise ValueError("Chemical identity mismatch: atom/bond counts differ")
    mapping = right.GetSubstructMatch(left)
    if len(mapping) != left.GetNumAtoms():
        raise ValueError("Chemical identity mismatch: non-isomorphic heavy-atom graph")
    inverse = [mapping.index(i) for i in range(len(mapping))]
    mol = Chem.RenumberAtoms(template, inverse)
    mol.RemoveAllConformers()
    mol.AddConformer(Chem.Conformer(observed.GetConformer()))
    return mol


def make_ligand(doc, rows, smiles, cache=None, ensemble=None):
    """OST and PoseBusters molecules for one ligand copy, with the route its graph came from.

    The OST molecule keeps connectivity from coordinates only. The PoseBusters molecule
    takes the SMILES template's chemistry, matched atom by atom onto a graph from the
    structure, another pose in the ensemble, or the coordinates. If that fails it is None
    and the reason is returned instead.
    """
    cache = {} if cache is None else cache
    coords = coordinate_mol(rows)
    names = [atom_name(r) for r in heavy_atoms(rows)]
    key = (ensemble, smiles, tuple(sorted(names)))
    template = template_mol(smiles)
    graph, provenance = native_graph(doc, rows)
    if provenance == "native_smiles_atom_names" and canonical_smiles(graph) != canonical_smiles(template):
        raise ValueError("Chemical identity mismatch between native and supplied SMILES")
    if graph is None and key in cache:
        peer_names, peer_graph = cache[key]
        graph = Chem.RenumberAtoms(peer_graph, [peer_names.index(n) for n in names])
        graph.RemoveAllConformers()
        graph.AddConformer(Chem.Conformer(coords.GetConformer()))
        provenance = "same_ensemble_atom_name_graph"
    if graph is None:
        graph = Chem.Mol(coords)
        rdDetermineBonds.DetermineConnectivity(graph)
        provenance = "coordinate_connectivity"
    ost_mol = Chem.Mol(coords)
    rdDetermineBonds.DetermineConnectivity(ost_mol)
    try:
        pb_mol = matched_template(template, graph)
        Chem.SanitizeMol(pb_mol)
        Chem.AssignStereochemistryFrom3D(pb_mol, replaceExistingTags=True)
        if canonical_smiles(pb_mol) != canonical_smiles(template):
            raise ValueError("Preparation changed chemical identity; PB not evaluated")
        if not np.array_equal(pb_mol.GetConformer().GetPositions(), coords.GetConformer().GetPositions()):
            raise ValueError("Preparation changed predicted coordinates")
        if [a.GetSymbol() for a in pb_mol.GetAtoms()] != [a.GetSymbol() for a in coords.GetAtoms()]:
            raise ValueError("Preparation changed atom correspondence")
    except (ValueError, RuntimeError) as exc:
        return ost_mol, None, provenance, str(exc)
    cache[key] = (names, Chem.Mol(pb_mol))
    return ost_mol, pb_mol, provenance, None


def write_molecule(mol, path):
    Chem.MolToMolFile(mol, str(path))
    reread = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    if reread is None or [a.GetSymbol() for a in reread.GetAtoms()] != [a.GetSymbol() for a in mol.GetAtoms()]:
        raise ValueError("SDF serialization failed")
    original = mol.GetConformer().GetPositions()
    serialized = reread.GetConformer().GetPositions()
    difference = float(np.max(np.abs(original - serialized)))
    if not np.isfinite(serialized).all() or not np.isfinite(difference) or difference > 5.00001e-5:
        raise ValueError(
            "SDF serialization changed coordinates beyond format precision"
        )
    return dict(atom_count=mol.GetNumAtoms(), max_coordinate_difference_angstrom=difference)
