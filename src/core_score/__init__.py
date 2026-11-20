"""CORE Score: agreement within an ensemble of predicted protein-ligand poses."""

__version__ = "0.6.1"

from core_score.pipeline import run
from core_score.analysis.core import core_score as score

__all__ = ["run", "score"]
