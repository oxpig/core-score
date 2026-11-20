"""Optional comparison of every prediction against its ensemble's ground-truth structure.
The ligand RMSD and lDDT-PLI are reported as they are, and never enter CORE."""

import math
from core_score.outputs.files import fingerprint, write_json
from core_score.analysis.pairs import run_pair_tasks
from core_score.inputs.prepare import prepare


def evaluate_groups(groups, output, *, workers=1, timeout=None, radius_fallback=True):
    """Ligand RMSD and lDDT-PLI of each prepared prediction to its reference. Ensembles
    without a reference are skipped."""
    groups = [group for group in groups if "reference" in group]
    tasks, records = [], []
    for group in groups:
        destination = output / "references" / fingerprint(group["ensemble"])[:20] / "reference.cif"
        reference = prepare(group["reference"], destination, timeout=timeout)
        for pose in group["poses"]:
            error = reference.get("error") or pose.get("error")
            if error:
                records.append(dict(ensemble=group["ensemble"], pose_id=pose["pose_id"], rmsd=None, error=error))
            else:
                tasks.append((dict(group, site="reference"), pose, reference))
    for row, detail in run_pair_tasks(tasks, workers, timeout=timeout, radius_fallback=radius_fallback, lddt_pli=True):
        records.append(dict(row, pose_id=row["pose_i"], detail=detail))
    write_json(output / "reference_details.json", records)
    return [reference_result(record) for record in records]


def reference_result(record):
    """A pose's RMSD and lDDT-PLI to the reference. A failed or ambiguous comparison leaves
    both blank, with the reason."""
    error = record.get("error")
    def known(value):
        return value if not error and value is not None and math.isfinite(value) else None
    return dict(ensemble=record["ensemble"], pose_id=record["pose_id"], rmsd=known(record.get("rmsd")),
                lddt_pli=known(record.get("lddt_pli")),
                error=(error["error"] if isinstance(error, dict) else error) or None)
