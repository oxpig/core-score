"""Build each site ensemble's RMSD matrix and calculate CORE at each threshold."""

import itertools
import numpy as np
from core_score.outputs.files import diagnostic

NEIGHBOURS = 3  # each pose is scored against its three closest other poses


def score_groups(groups, rows, thresholds=(2.0,)):
    """One score per site ensemble and threshold T. An ensemble with a missing or failed pair is not scored."""
    scores = []
    for group in groups:
        ids = [pose["pose_id"] for pose in group["poses"]]
        values, error = dict.fromkeys(thresholds), group.get("error")
        if not error:
            try:
                key = (group["ensemble"], group["site"])
                matrix = build_matrix(ids, [r for r in rows if (r["ensemble"], r["site"]) == key])
                values = {threshold: core_score(matrix, threshold) for threshold in thresholds}
            except ValueError as exc:
                error = diagnostic("scoring", exc)
        scores += [
            dict(ensemble=group["ensemble"], site=group["site"], name=group["name"], threshold_angstrom=threshold,
                 core_score=value, error=error["error"] if error else None)
            for threshold, value in values.items()
        ]
    return scores


def build_matrix(pose_ids, rows):
    """Pairwise ligand-RMSD matrix. Raises unless every pair has exactly one usable RMSD;
    a missing or failed comparison is never filled in."""
    index = {pose_id: i for i, pose_id in enumerate(pose_ids)}
    matrix = np.full((len(pose_ids), len(pose_ids)), np.nan)
    np.fill_diagonal(matrix, 0.0)
    for row in rows:
        if row["n_assigned"] == 1 and not row["error"]:
            i, j = index[row["pose_i"]], index[row["pose_j"]]
            matrix[i, j] = matrix[j, i] = row["rmsd"]
    missing = [(a, b) for (i, a), (j, b) in itertools.combinations(enumerate(pose_ids), 2)
               if not np.isfinite(matrix[i, j])]
    if missing:
        raise ValueError(f"No usable ligand RMSD for pairs {missing}; CORE needs every pair")
    return matrix


def core_score(matrix, threshold=2.0, neighbours=NEIGHBOURS):
    """CORE Score of an ensemble from its complete pairwise ligand-RMSD matrix.

    Each pose's RMSDs to its `neighbours` closest other poses are capped at `threshold`
    and divided by it, and their mean is that pose's penalty: zero when those poses
    reproduce it exactly, one when none of them lies within the threshold. CORE is one
    minus the mean penalty over all poses, so it needs at least `neighbours` + 1 poses.
    """
    distances = np.array(matrix, dtype=float)
    if len(distances) <= neighbours:
        raise ValueError(f"CORE needs at least {neighbours + 1} poses; this ensemble has {len(distances)}")
    np.fill_diagonal(distances, np.inf)
    closest = np.sort(distances, axis=1)[:, :neighbours]
    return float(1.0 - np.mean(np.minimum(closest, threshold) / threshold))
