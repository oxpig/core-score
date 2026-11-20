"""Write the two files a user reads: core_scores.csv, one row per complex, and
predictions.csv, one row per prediction. Everything behind them is in details/."""

from core_score.outputs.files import write_csv


def write_report(sites, scores, pb_records, references, output):
    """Write both files and return each threshold's CORE Scores, for the plots.

    A complex is an ensemble, or one ligand site of it. `sites` are the site ensembles,
    `scores` their CORE at each threshold, `pb_records` PoseBusters' verdicts (None when it
    was skipped) and `references` each pose's RMSD and lDDT-PLI to its reference.
    """
    validity = None if pb_records is None else {(r["ensemble"], r["pose_id"]): r for r in pb_records}
    measured = {(r["ensemble"], r["pose_id"]): r for r in references}
    core = core_columns(scores)
    sites = sorted(sites, key=lambda site: rank(core[site["name"]]))
    with_confidence = any("confidence" in site["poses"][0] for site in sites)
    write_csv(output / "core_scores.csv",
              without_empty_notes([complex_row(site, core[site["name"]], validity) for site in sites]))
    write_csv(output / "predictions.csv",
              without_empty_notes([pose_row(site, pose, core[site["name"]], validity, measured, with_confidence)
                                   for site in sites for pose in site["poses"]]))
    by_threshold = {}
    for score in scores:
        by_threshold.setdefault(score["threshold_angstrom"], []).append(score["core_score"])
    return by_threshold


def core_columns(scores):
    """Each complex's CORE as columns: core_score, or core_score_<T>A for several thresholds,
    with the reason when it is missing."""
    thresholds = list(dict.fromkeys(score["threshold_angstrom"] for score in scores))
    columns = {}
    for score in scores:
        name = "core_score" if len(thresholds) == 1 else f"core_score_{score['threshold_angstrom']:g}A"
        columns.setdefault(score["name"], {})[name] = score["core_score"]
        columns[score["name"]]["_error"] = score["error"]
    return columns


def rank(core):
    """Highest CORE first, by the first threshold; complexes without CORE last."""
    first = next(value for key, value in core.items() if key != "_error")
    return (first is None, -(first or 0))


def scores_only(core):
    return {key: value for key, value in core.items() if key != "_error"}


def without_empty_notes(rows):
    """Leave out the note column when nothing needed one."""
    if not any(row["note"] for row in rows):
        for row in rows:
            del row["note"]
    return rows


def complex_row(site, core, validity):
    """CORE, how many predictions the complex has and, with PoseBusters, how many are valid."""
    poses = site["poses"]
    row = dict(ensemble=site["name"], **scores_only(core), prediction_count=len(poses))
    if validity is not None:
        row["valid_prediction_count"] = sum(validity.get((p["ensemble"], p["pose_id"]), {}).get("pb_valid") is True
                                            for p in poses)
    row["note"] = core["_error"]
    return row


def pose_row(site, pose, core, validity, measured, with_confidence):
    """The pose's complex and its CORE, its confidence, PoseBusters' verdict and its scores
    against the reference. Columns that do not apply are left out, and a blank is unknown."""
    key = (pose["ensemble"], pose["pose_id"])
    row = dict(ensemble=site["name"], prediction=pose["pose_id"], **scores_only(core))
    if with_confidence:
        row["confidence"] = pose.get("confidence")
    record = validity.get(key) if validity is not None else None
    if validity is not None:
        # Which checks failed is in details/posebusters.json.
        row["pb_valid"] = record["pb_valid"] if record else None
    scores = measured.get(key, {})
    if measured:
        row.update(rmsd=scores.get("rmsd"), lddt_pli=scores.get("lddt_pli"))
    row["note"] = message(pose.get("error")) or (record or {}).get("error") or scores.get("error")
    return row


def message(error):
    """The one-line message of a diagnostic record, or of a plain error string."""
    return (error["error"] if isinstance(error, dict) else error) or None
