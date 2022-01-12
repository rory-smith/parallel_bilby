import json
import os
import random

import pytest
from mpi4py import MPI
from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool


def _func(args):
    a, b = args
    return a + b


def _pool_test(pool):
    a = [random.random() for _ in range(100)]
    b = [random.random() for _ in range(100)]

    args = list(zip(a, b))

    return pool.map(_func, args)


@pytest.mark.mpi
def test_dill():
    with MPIPool(use_dill=True) as pool:
        if pool.is_master():
            results = _pool_test(pool)
            assert all(0 <= x <= 2 for x in results)


@pytest.mark.mpi
def test_timer():
    with MPIPool(time_mpi=True, timing_interval=0.001) as pool:
        if pool.is_master():
            results = _pool_test(pool)
            assert all(0 <= x <= 2 for x in results)

    # Need to run tests outside of pool context, but only on master
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        timing_file = "mpi_worker_timing.json"
        assert os.path.exists(timing_file)

        try:
            with open(timing_file, "r") as f:
                data = json.load(f)
        finally:
            os.remove(timing_file)

        assert type(data) == list
        assert len(data) >= 1
        assert type(data[0]) == dict

        expected_keys = [
            "master_serial",
            "mpi_recv",
            "compute",
            "mpi_send",
            "barrier",
            "walltime",
        ]
        for entry in expected_keys:
            assert entry in data[0].keys()
