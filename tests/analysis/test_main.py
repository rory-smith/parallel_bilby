import os.path
import shutil
import unittest

import dill
import pytest
from mpi4py import MPI
from parallel_bilby import analysis, generation
from utils import mpi_master

OUTDIR = "tests/test_files/out_main_test/"


class MainTest(unittest.TestCase):
    @mpi_master
    def setUp(self):
        generation.generate_runner(
            ["tests/test_files/fast_test.ini", "--outdir", OUTDIR]
        )

    @mpi_master
    def tearDown(self):
        shutil.rmtree(OUTDIR)

    @pytest.mark.mpi
    def test_max_its(self):
        its = 5
        # Run analysis
        exit_reason = analysis.analysis_runner(
            data_dump=os.path.join(OUTDIR, "data/fast_injection_data_dump.pickle"),
            outdir=os.path.join(OUTDIR, "result"),
            label="fast_injection_0",
            max_its=its,
        )

        check_master_value(exit_reason, 1)

        # Only check the pickle file on master
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(
                os.path.join(
                    OUTDIR, "result/fast_injection_0_checkpoint_resume.pickle"
                ),
                "rb",
            ) as f:
                resume_file = dill.load(f)

            # The code runs 2 more iterations than specified
            assert resume_file.it == its + 2

    @pytest.mark.mpi
    def test_max_time(self):
        time = 0.1
        # Run analysis
        exit_reason = analysis.analysis_runner(
            data_dump=os.path.join(OUTDIR, "data/fast_injection_data_dump.pickle"),
            outdir=os.path.join(OUTDIR, "result"),
            label="fast_injection_0",
            max_run_time=time,
        )

        check_master_value(exit_reason, 2)

    @pytest.mark.mpi
    def test_resume(self):
        # Run analysis for 5 iterations
        exit_reason = analysis.analysis_runner(
            data_dump=os.path.join(OUTDIR, "data/fast_injection_data_dump.pickle"),
            outdir=os.path.join(OUTDIR, "result"),
            label="fast_injection_0",
            max_its=5,
        )
        # Sanity check: make sure the run stopped because of max iterations
        check_master_value(exit_reason, 1)

        exit_reason = analysis.analysis_runner(
            data_dump=os.path.join(OUTDIR, "data/fast_injection_data_dump.pickle"),
            outdir=os.path.join(OUTDIR, "result"),
            label="fast_injection_0",
        )
        check_master_value(exit_reason, 0)


@mpi_master
def check_master_value(test_value, expected_value):
    assert test_value == expected_value
