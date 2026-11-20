"""One command: load inputs and run the analysis."""

import argparse
import math
import sys
from pathlib import Path
from core_score.pipeline import run


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def worker_count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    p = argparse.ArgumentParser(
        prog="core-score",
        description="Prepare structures, run PoseBusters, compare predictions and calculate CORE.",
        epilog="Example: core-score core_score_input.csv --out results/ --workers 4. Make the CSV with core-score-csv.",
    )
    p.add_argument("inputs", metavar="INPUT", nargs="+", type=Path,
                   help="CSV of ensembles, or structure files forming one ensemble")
    p.add_argument("--out", type=Path, required=True, help="Results directory")
    p.add_argument("--ligand", help="The ligand as a SMILES or an SDF file, for structure files; a CSV sets its own")
    p.add_argument("--ligand-chain", dest="ligand_chains", action="append",
                   help="Ligand chain for structure files; repeat for identical copies")
    p.add_argument("--workers", type=worker_count, default=1,
                   help="Simultaneous OST comparisons (default: 1)")
    p.add_argument("--timeout", type=positive, default=None,
                   help="Optional seconds per OST comparison; no automatic timeout retries (default: no timeout)")
    p.add_argument("--no-radius-fallback", action="store_true",
                   help="Disable eligible binding-site retries at 6 and 8 angstroms")
    p.add_argument("--skip-posebusters", action="store_true", help="Do not run PoseBusters checks")
    p.add_argument("--reference", type=Path,
                   help="Ground-truth structure for structure files; a CSV sets its own")
    p.add_argument("--rmsd-thresholds", type=positive, nargs="+", default=[2.0],
                   help="Thresholds T in angstroms at which CORE is calculated (default: 2)")
    p.add_argument("--plot", action="store_true",
                   help="Also save core_scores.png and core_score_distribution.png (needs matplotlib)")
    return p


def main(argv=None):
    words = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(words or ["--help"])
    try:
        summary = run(
            args.inputs,
            args.out,
            ligand=args.ligand,
            ligand_chains=args.ligand_chains,
            reference=args.reference,
            workers=args.workers,
            timeout=args.timeout,
            radius_fallback=not args.no_radius_fallback,
            skip_posebusters=args.skip_posebusters,
            thresholds=args.rmsd_thresholds,
            plot=args.plot,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"core-score: {exc}", file=sys.stderr)
        return 2
    print(
        f"Completed {summary['completed_comparisons']}/{summary['comparison_count']} "
        f"pair comparisons; calculated {summary['score_count']} CORE scores."
    )
    print(f"Results saved to {args.out}")
    return int(any(summary["status"].values()))
