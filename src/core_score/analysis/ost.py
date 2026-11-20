"""Run OpenStructure's command-line actions. `ost` is found on PATH, and its chemical
library through OST's own discovery or OST_COMPOUNDS_CHEMLIB."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from core_score.analysis import transport


INSTALL = "conda install -c conda-forge -c bioconda openstructure"
SYMMETRY_RETRY_LIMITS = (100_000, 1_000_000)


def require_ost():
    """Stop before any work when OpenStructure is not installed, saying how to install it."""
    if shutil.which("ost") is None:
        raise RuntimeError(f"OpenStructure's ost command was not found. Install it with: {INSTALL}")


class OSTError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def execute(arguments, timeout=None):
    """Run `ost <arguments> -o <file>` and return its JSON output with the execution record."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "ost.json"
        try:
            execution = transport.run(arguments, output, timeout)
        except subprocess.TimeoutExpired as exc:
            event = dict(event="timeout", arguments=list(map(str, arguments)), timeout=timeout,
                         attempt=1, retry=False, error=str(exc))
            transport.audit(event)
            raise OSTError(f"TimeoutExpired: {exc}", dict(timeout_retries=[event])) from exc
        execution["timeout_retries"] = []
        if not output.exists():
            raise OSTError(execution["stderr"].strip() or
                           f"OST exited with code {execution['returncode']}", execution)
        try:
            result = json.loads(output.read_text())
        except (ValueError, OSError) as exc:
            raise OSTError(f"Invalid OST output: {exc}", execution) from exc
        return result, execution


def compare_ligands(model, reference, timeout=None, radius=4.0, lddt_pli=False):
    """`ost compare-ligand-structures` result for one model/reference pair of prepared structures.
    lDDT-PLI is only computed when asked for, since CORE uses RMSD alone."""
    arguments = ["compare-ligand-structures", "-m", model["path"]]
    if model["ligand_sdfs"]:
        arguments += ["-ml", *model["ligand_sdfs"]]
    arguments += ["-r", reference["path"]]
    if reference["ligand_sdfs"]:
        arguments += ["-rl", *reference["ligand_sdfs"]]
    arguments += ["--fault-tolerant", "--rmsd", "--full-bs-search", "--full-results"]
    if lddt_pli:
        arguments += ["--lddt-pli", "--lddt-pli-add-mdl-contacts"]
    if radius != 4.0:
        arguments += ["--radius", radius]
    symmetry_attempts = []
    for limit in (None, *SYMMETRY_RETRY_LIMITS):
        command = arguments if limit is None else [*arguments, "--max-symmetries", str(limit)]
        result, execution = execute(command, timeout)
        if execution["returncode"] or result.get("status") != "SUCCESS":
            result["status"] = "FAILURE"
            result.setdefault("exception", execution["stderr"][-2000:] or "OST did not report success")
        limited = result.get("status") == "SUCCESS" and symmetry_limit_reached(result)
        symmetry_attempts.append(dict(max_symmetries=limit, limit_reached=limited, execution=dict(execution)))
        if not limited:
            break
    if len(symmetry_attempts) > 1:
        execution["symmetry_attempts"] = symmetry_attempts
        transport.audit(dict(event="symmetry_limit_retry", attempts=symmetry_attempts))
    result["_execution"] = execution
    return result


def symmetry_limit_reached(result):
    """Retry only OST's explicit symmetry-limit reason, never a graph/pocket error."""
    for metric in ("rmsd", "lddt_pli"):
        scores = result.get(metric, {})
        for field in ("model_ligand_unassigned_reason", "reference_ligand_unassigned_reason"):
            if any(isinstance(reason, (list, tuple)) and reason and reason[0] == "symmetries"
                   for reason in scores.get(field, {}).values()):
                return True
    return False


def superpose(model_path, reference_path, timeout=None):
    """OST rigid superposition of matched protein C-alpha atoms, model onto reference."""
    result, execution = execute(
        ["compare-structures", "-m", model_path, "-r", reference_path,
         "--rigid-scores", "--fault-tolerant"], timeout
    )
    if execution["returncode"] or result.get("status") != "SUCCESS" or result.get("transform") is None:
        raise OSTError(
            f"OST rigid superposition failed (status {result.get('status')}); chains may not map",
            dict(execution, ost_results=result),
        )
    return dict(transform=np.asarray(result["transform"], dtype=float),
                rmsd=result.get("rmsd"), chain_mapping=result.get("chain_mapping") or {})
