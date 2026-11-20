"""Small, dependency-light tests of the scientific output contract."""

import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gemmi
from core_score.analysis.pairs import pair_job, summarize
from core_score.inputs.prepare import prepare_pose
from core_score.inputs.structures import filter_records, ligand_residues, read_document


class SummaryTests(unittest.TestCase):
    def test_assigned_values_are_not_replaced_by_a_better_unassigned_match(self):
        payload = dict(
            status="SUCCESS",
            rmsd=dict(
                assigned_scores=[
                    dict(score=1.23456789012345, bb_rmsd=0.2, lddt_lp=0.7)
                ],
                full_results=[dict(score=0.01)],
            ),
            lddt_pli=dict(assigned_scores=[dict(score=0.81234567890123)]),
        )
        original = copy.deepcopy(payload)
        row = summarize(payload)
        self.assertEqual(row["rmsd"], 1.23456789012345)
        self.assertEqual(row["lddt_pli"], 0.81234567890123)
        self.assertEqual(payload, original)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        stream.seek(0)
        decoded = next(csv.DictReader(stream))
        for metric in ("rmsd", "lddt_pli", "bb_rmsd", "lddt_lp"):
            self.assertEqual(float(decoded[metric]), row[metric])

    def test_missing_assignment_is_not_zero(self):
        row = summarize(dict(status="SUCCESS", rmsd=dict(assigned_scores=[])))
        self.assertEqual(row["n_assigned"], 0)
        self.assertIsNone(row["rmsd"])

    def test_repeated_copies_have_no_single_value(self):
        row = summarize(
            dict(status="SUCCESS", rmsd=dict(assigned_scores=[dict(score=1.0), dict(score=3.0)]))
        )
        self.assertEqual(row["n_assigned"], 2)
        self.assertIsNone(row["rmsd"])

    def test_failure_cannot_leak_partial_values_as_success(self):
        row = summarize(
            dict(
                status="FAILURE",
                exception="mapping failed",
                rmsd=dict(assigned_scores=[dict(score=0.1)]),
            )
        )
        self.assertIsNone(row["rmsd"])
        self.assertEqual(row["error"], "mapping failed")

    def test_timeout_is_explicit(self):
        import subprocess

        prepared = dict(path="model.cif", ligand_sdfs=[], copies=1)
        pose = lambda name: dict(pose_id=name, path=name, prepared=prepared)
        with patch(
            "core_score.analysis.transport.run",
            side_effect=subprocess.TimeoutExpired("ost", 1),
        ):
            row, detail = pair_job((dict(ensemble="e", site="1"), pose("a"), pose("b")))
        self.assertEqual(row["n_assigned"], 0)
        self.assertIn("TimeoutExpired", row["error"])


class PreparationTests(unittest.TestCase):
    CIF = """data_example
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
ATOM 1 C CA A 1.23456789 -2.00000001 3.0 0.50
HETATM 2 C "C 1"
B 4.00000001 5.0 6.0 1.00
HETATM 3 N N1 C 7.0 8.0 9.0 1.00
HETATM 4 O O1 X 10.0 11.0 12.0 1.00
"""

    def test_wrapped_quoted_records_preserve_every_retained_field_and_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.cif"
            source.write_text(self.CIF)
            original = source.read_bytes()
            destination = Path(tmp) / "prepared.cif"
            doc = read_document(source)
            kept = {key for key in ligand_residues(doc) if key[0] in ("B", "C")}
            filter_records(doc, kept, destination)
            before = (
                gemmi.cif.read_file(str(source.resolve()))
                .sole_block()
                .find_mmcif_category("_atom_site.")
            )
            after = (
                gemmi.cif.read_file(str(destination))
                .sole_block()
                .find_mmcif_category("_atom_site.")
            )
            self.assertEqual(
                [list(r) for r in after], [list(r) for r in before if r[4] != "X"]
            )
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(set(ligand_residues(read_document(destination))), kept)

    def test_missing_chain_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.cif"
            source.write_text(self.CIF)
            with self.assertRaisesRegex(ValueError, "absent"):
                prepare_pose(
                    dict(path=str(source), ligand_chains=["Z"]), Path(tmp) / "out.cif"
                )


if __name__ == "__main__":
    unittest.main()
