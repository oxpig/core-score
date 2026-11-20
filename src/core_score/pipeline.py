"""Run the analysis stages in order and save every dataset."""

import os
import shutil
from importlib.metadata import version
from pathlib import Path
from core_score.inputs.inputs import load_inputs
from core_score.outputs.files import write_json
from core_score.analysis.pairs import compare_groups
from core_score.analysis.reference import evaluate_groups
from core_score.analysis.posebusters import run_posebusters
from core_score.inputs.prepare import prepare_groups
from core_score.outputs.report import write_report
from core_score.analysis.core import score_groups
from core_score.inputs.sites import prepare_site_groups
from core_score.analysis.ost import require_ost
from core_score.analysis import transport
from core_score import __version__


def run(inputs, out, *, ligand=None, ligand_chains=None, reference=None, **options):
    """Run CORE Score from Python, exactly as the core-score command does.

    `inputs` is a CSV file or a list of structure files. The other settings match the
    command-line options, for example workers=4 or thresholds=(1.0, 2.0).
    Results are saved under `out`, and the completion counts are returned.

        import core_score
        core_score.run("core_score_input.csv", "results/", workers=8)
    """
    inputs = [inputs] if isinstance(inputs, (str, os.PathLike)) else inputs
    groups = load_inputs(inputs, ligand=ligand, ligand_chains=ligand_chains, reference=reference)
    return execute(groups, out, **options)


def execute(
    groups,
    output,
    *,
    workers=1,
    timeout=None,
    radius_fallback=True,
    skip_posebusters=False,
    thresholds=(2.0,),
    plot=False,
):
    """Run the full analysis on loaded ensembles and return completion counts.

    The files a user reads go in `output`, and the evidence behind them in `output/details`.
    The working copies of every structure are deleted at the end. Poses are compared with a
    reference wherever their ensemble has one.
    """
    require_ost()
    evaluate_reference = any("reference" in group for group in groups)
    output = Path(output).resolve()
    thresholds = tuple(map(float, thresholds))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new or empty output directory")
    details = output / "details"
    details.mkdir(parents=True, exist_ok=True)
    write_json(details / "run.json", dict(
        package_version=__version__, posebusters_version=version("posebusters"), timeout=timeout,
        radius_fallback=radius_fallback, workers=workers, skip_posebusters=skip_posebusters,
        thresholds=thresholds, evaluate_reference=evaluate_reference, plot=plot,
        ligand_connectivity="smiles_matched_explicit_connectivity_v1",
        ost_mode=transport.mode(), ost_max_timeout_retries=0,
        structures_cached=False, posebusters_energy_threads=1,
    ))

    with transport.session(details):
        print("Preparing structures and ligand chemistry", flush=True)
        prepared = prepare_groups(groups, details, timeout=timeout)
        write_json(details / "ligand_preparation.json", [
            dict(ensemble=pose["ensemble"], pose_id=pose["pose_id"], error=pose.get("error"),
                 route=pose.get("prepared", {}).get("route"),
                 copies=[{key: copy.get(key) for key in ("copy_id", "provenance", "pb_error", "coordinate_audit")}
                         for copy in pose.get("prepared", {}).get("copy_records", [])])
            for group in prepared for pose in group["poses"]
        ])
        pb_records = None
        if not skip_posebusters:
            print("Running PoseBusters dock checks", flush=True)
            pb_records = run_posebusters(prepared, details)
        sites = prepare_site_groups(prepared, details, timeout=timeout)
        print(f"Comparing predictions with {workers} simultaneous OST worker(s)", flush=True)
        rows = compare_groups(sites, details, workers=workers, timeout=timeout, radius_fallback=radius_fallback)
        scores = score_groups(sites, rows, thresholds)
        references = []
        if evaluate_reference:
            print("Comparing predictions with their references", flush=True)
            references = evaluate_groups(prepared, details, workers=workers, timeout=timeout, radius_fallback=radius_fallback)
        scores_by_threshold = write_report(sites, scores, pb_records, references, output)
        if plot:
            from core_score.outputs.plot import plot_core_scores
            plot_core_scores(scores_by_threshold, output)

        for working in ("prepared", "references"):
            shutil.rmtree(details / working, ignore_errors=True)
        return summarize_run(scores, rows, sites, references)


def summarize_run(scores, rows, groups, references):
    """Completion counts; the measurements themselves are in the saved datasets."""
    ensembles = {(score["ensemble"], score["site"]) for score in scores}
    scored = {(score["ensemble"], score["site"]) for score in scores if score["core_score"] is not None}
    failed_comparisons = sum(bool(row["error"]) for row in rows)
    return {
        "comparison_count": len(rows),
        "completed_comparisons": len(rows) - failed_comparisons,
        "score_count": len(scored),
        "status": {
            "incomplete_scores": len(ensembles - scored),
            "failed_comparisons": failed_comparisons,
            "unresolved_ensembles": sum(bool(group.get("error")) for group in groups),
            "unknown_reference_scores": sum(result["rmsd"] is None for result in references),
        },
    }
