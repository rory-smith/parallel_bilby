import numpy as np
import pytest
from parallel_bilby.analysis.likelihood import reorder_loglikelihoods


@pytest.mark.mpi_skip
def test_reorder_loglikelihoods():
    """
    Test that the reorder_loglikelihoods method reorders correctly.
    """
    sorted_s = np.array([[1, 2, 3, 4]]).T
    unsorted_s = np.array([[3, 1, 4, 2]]).T
    unsorted_l = np.array([[8, 6, 9, 7]]).T
    sorted_l = np.array([[6, 7, 8, 9]]).T

    assert np.all(sorted_l == reorder_loglikelihoods(unsorted_l, unsorted_s, sorted_s))