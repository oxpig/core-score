"""PoseBusters dock checks for every pose. The results are saved for use downstream, such as
choosing a top-ranked prediction. CORE uses every pose whatever its result. Unknown never
counts as a pass."""

from copy import deepcopy
import logging
from core_score.outputs.files import clean, diagnostic, write_json


def run_posebusters(groups, output):
    from posebusters import PoseBusters

    buster = PoseBusters(config="dock", max_workers=0)
    buster.config = deepcopy(buster.config)
    for module in buster.config["modules"]:
        if module.get("function") == "energy_ratio":
            module.setdefault("parameters", {})["num_threads"] = 1
    checks = dock_checks(buster)
    records = [check_pose(buster, checks, pose) for group in groups for pose in group["poses"]]
    write_json(output / "posebusters.json", records)
    return records


def dock_checks(buster):
    """Names of the binary checks in PoseBusters' dock configuration."""
    return [
        normalized(module.get("rename_outputs", {}).get(name, name))
        for module in buster.config["modules"]
        for name in module.get("chosen_binary_test_output", [])
    ]


def normalized(name):
    return name.lower().replace(" ", "_")


def aggregate(values):
    """False if any check failed, None if any is unknown (or there are none), otherwise True."""
    values = list(values)
    if any(v is False for v in values):
        return False
    if not values or any(v is None for v in values):
        return None
    return True


def check_pose(buster, checks, pose):
    """A pose is valid only if every ligand copy passes every dock check. Each check's result
    for the pose combines its copies the same way."""
    record = dict(ensemble=pose["ensemble"], pose_id=pose["pose_id"], pb_valid=None,
                  checks=dict.fromkeys(checks))
    if pose.get("error"):
        return dict(record, error="The prediction could not be prepared")
    prepared = pose["prepared"]
    copies = [check_copy(buster, checks, copy, prepared["receptor"]) for copy in prepared["copy_records"]]
    combined = {name: aggregate(copy.get("checks", {}).get(name) for copy in copies) for name in checks}
    verdict = aggregate(copy["pb_valid"] for copy in copies)
    result = dict(record, pb_valid=verdict, checks=combined, copies=copies)
    if verdict is None:
        missing = [name for name, value in combined.items() if value is None]
        reasons = [copy["error"] for copy in copies if copy.get("error")]
        result["error"] = "PoseBusters result unknown: " + "; ".join(reasons or missing or ["no checks returned"])
    return result


def check_copy(buster, checks, copy, receptor):
    record = dict(copy_id=copy["copy_id"], preparation=copy["provenance"], pb_valid=None)
    if copy["pb_error"]:
        return dict(record, error=copy["pb_error"])
    messages = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = lambda item: messages.append(item.getMessage())
    logger = logging.getLogger("posebusters")
    logger.addHandler(handler)
    try:
        report = buster.bust(copy["pb_sdf"], mol_cond=receptor, full_report=True)
    except Exception as exc:
        return dict(record, **diagnostic("posebusters", exc))
    finally:
        logger.removeHandler(handler)
    tests = {normalized(k): clean(v) for k, v in report.iloc[0].items()}
    results = {name: tests.get(name) if isinstance(tests.get(name), bool) else None for name in checks}
    result = dict(record, pb_valid=aggregate(results.values()), checks=results, tests=tests)
    if messages:
        result["warnings"] = list(dict.fromkeys(messages))
        if result["pb_valid"] is None:
            result["error"] = "; ".join(result["warnings"])
    return result

