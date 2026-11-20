import numpy as np
import pytest
from core_score.inputs.sites import assign_sites


def test_site_assignment_does_not_accept_nonconvergence():
    copies = {"B": np.array([0.0, 0.0, 0.0]), "C": np.array([10.0, 0.0, 0.0])}
    with pytest.raises(ValueError, match="converge"):
        assign_sites({"a": copies}, 2, max_iter=1)
