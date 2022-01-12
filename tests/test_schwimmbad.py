import random

import pytest
from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool


@pytest.mark.mpi
def test_dill():
    def func(args):
        a, b = args
        return a + b

    with MPIPool(use_dill=True) as pool:
        if pool.is_master():
            a = [random.random() for _ in range(100)]
            b = [random.random() for _ in range(100)]

            args = list(zip(a, b))

            results = pool.map(func, args)

            assert all(0 <= x <= 2 for x in results)
