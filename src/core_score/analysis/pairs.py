"""Compare every pair of poses within each site ensemble with OST and save the measurements."""

import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from core_score.analysis import ost
from core_score.outputs.files import diagnostic, write_csv, write_json


def compare_groups(groups, output, *, workers=1, timeout=None, radius_fallback=True):
    results = run_pair_tasks(pair_tasks(groups), workers, timeout=timeout, radius_fallback=radius_fallback)
    results.sort(key=lambda result: tuple(result[0][k] for k in ("ensemble", "site", "pose_i", "pose_j")))
    rows = [row for row, _ in results]
    write_csv(output / "pairs.csv", rows)
    failures = [detail for row, detail in results if row["error"]]
    if failures:
        write_json(output / "failed_pairs.json", failures)
    return rows


def pair_tasks(groups):
    """Every pair of poses within each site ensemble; earlier poses are the OST model."""
    return [
        (group, pose_i, pose_j)
        for group in groups
        if not group.get("error")
        for pose_i, pose_j in itertools.combinations(group["poses"], 2)
    ]


def run_pair_tasks(tasks, workers=1, **options):
    """(row, detail) for each task, in task order, using up to `workers` processes."""
    job = partial(pair_job, **options)
    if workers == 1:
        return [job(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(job, tasks))


def pair_job(task, *, timeout=None, radius_fallback=True, lddt_pli=False):
    group, pose_i, pose_j = task
    ids = dict(ensemble=group["ensemble"], site=group["site"],
               pose_i=pose_i["pose_id"], pose_j=pose_j["pose_id"])
    try:
        row, detail = compare_pair(pose_i, pose_j, timeout=timeout, radius_fallback=radius_fallback, lddt_pli=lddt_pli)
    except Exception as exc:
        detail = diagnostic("comparison", exc, inputs=[pose_i["path"], pose_j["path"]])
        row = dict(n_assigned=0, error=detail["error"])
    return dict(ids, **row), dict(ids, **detail)


def compare_pair(pose_i, pose_j, *, timeout=None, radius_fallback=True, lddt_pli=False):
    """OST scores for one pair, retried at 6 and then 8 A after a binding-site failure at 4 A.

    The RMSD metrics come from the radius that scored. lDDT-PLI always stays from 4 A.
    """
    model, reference = pose_i["prepared"], pose_j["prepared"]
    # Recovery needs one ligand copy per side. Ensembles are split by site before comparison.
    retry = radius_fallback and model["copies"] == reference["copies"] == 1
    results, attempts = [], []
    for radius in (4.0, 6.0, 8.0) if retry else (4.0,):
        result = ost.compare_ligands(model, reference, timeout, radius, lddt_pli)
        failed = result["status"] == "FAILURE"
        assigned = [] if failed else result.get("rmsd", {}).get("assigned_scores", [])
        score = assigned[0].get("score") if len(assigned) == 1 else None
        reason = result["exception"] if failed else result.get("rmsd", {})
        results.append(result)
        attempts.append(dict(radius=radius, score=score, reason=reason))
        if usable(score) or not binding_site_failure(reason):
            break
    recovered = len(attempts) > 1 and usable(attempts[-1]["score"])
    result = dict(results[0])
    if recovered:
        result["rmsd"] = results[-1]["rmsd"]
    row = summarize(result)
    if row["n_assigned"] == 0 and not row["error"]:
        row["error"] = "OST assigned no ligand pair; see the saved OST results"
    scored = attempts[-1] if recovered else attempts[0]
    row.update(
        standard_rmsd=attempts[0]["score"] if usable(attempts[0]["score"]) else None,
        binding_site_radius_angstrom=scored["radius"] if usable(scored["score"]) else None,
        radius_fallback_used=recovered,
    )
    detail = dict(error=row["error"], source_pose_i=pose_i["path"], source_pose_j=pose_j["path"],
                  radius_attempts=attempts, ost_results=results)
    return row, detail


def summarize(result):
    """OST's scores for the pair, blank unless exactly one ligand pair was assigned."""
    failed = result["status"] == "FAILURE"
    assigned = [] if failed else [
        s for s in result.get("rmsd", {}).get("assigned_scores", []) if s.get("score") is not None
    ]
    contacts = [] if failed else [
        s["score"] for s in result.get("lddt_pli", {}).get("assigned_scores", []) if s.get("score") is not None
    ]
    single = assigned[0] if len(assigned) == 1 else {}
    return dict(
        n_assigned=len(assigned),
        rmsd=single.get("score"),
        bb_rmsd=single.get("bb_rmsd"),
        lddt_lp=single.get("lddt_lp"),
        lddt_pli=contacts[0] if len(contacts) == 1 else None,
        error=str(result.get("exception") or result.get("traceback") or "OST failed") if failed else "",
    )


def usable(score):
    """A finite, non-negative RMSD."""
    return score is not None and math.isfinite(float(score)) and float(score) >= 0


def binding_site_failure(result):
    """Whether OST's reasons name a binding-site failure (not a graph or general failure)."""
    if isinstance(result, dict):
        return any(binding_site_failure(v) for v in result.values())
    if isinstance(result, (list, tuple)):
        return any(binding_site_failure(v) for v in result)
    return result in ("target_binding_site", "model_binding_site", "binding_site")
