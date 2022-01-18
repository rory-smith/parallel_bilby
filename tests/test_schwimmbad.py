import json
import os
import random
import time

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
@pytest.mark.parametrize("enabled", [False, True])  # False case tests the null timer
def test_timer(enabled):
    if enabled:
        interval = 1.0e-3
    else:
        interval = None

    sleep_time = 1.0

    with MPIPool(time_mpi=enabled, timing_interval=interval) as pool:
        if pool.is_master():
            results = _pool_test(pool)
            # Trigger the "master_serial" timer
            time.sleep(sleep_time)
            results = _pool_test(pool)
            assert all(0 <= x <= 2 for x in results)

    if enabled:
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

            assert sum(step["master_serial"] for step in data) == pytest.approx(
                sleep_time, abs=0.1
            )


@pytest.mark.mpi
def test_enabled():
    with MPIPool() as pool:
        assert pool.enabled()


@pytest.mark.mpi
def test_without_context():
    # Open the pool manually
    pool = MPIPool()

    # Break workers out of first wait loop
    if pool.is_master():
        pool.kill_workers()

    results = _pool_test(pool)
    if pool.is_master():
        assert all(0 <= x <= 2 for x in results)

    # Close the pool manually
    pool.close()
    assert pool.pool_open is False
