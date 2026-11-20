"""Shared output writers and the diagnostic record kept for every failure."""

import csv
import hashlib
import json
import math
import traceback
from pathlib import Path


def clean(value):
    """JSON-safe copy: paths and NumPy values become plain types, NaN/inf and pandas NA become None."""
    if type(value).__name__ in ("NAType", "NaTType"):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v)) if isinstance(v, (dict, list, tuple)) else clean(v)
                             for k, v in row.items()})


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def diagnostic(stage, exc, **context):
    """What failed, where, and the evidence an exception carried (such as an OST command and its output)."""
    return dict(
        stage=stage,
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
        evidence=getattr(exc, "evidence", None),
        **context,
    )


def pose_directory(output, pose):
    """Where a pose's prepared files are written: prepared/<hash of ensemble and pose ID>/."""
    return Path(output) / "prepared" / fingerprint((pose["ensemble"], pose["pose_id"]))[:20]
