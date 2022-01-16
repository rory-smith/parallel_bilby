import os.path
import shutil
import unittest

import bilby
import dill
import pytest
from mpi4py import MPI
from parallel_bilby import analysis, generation
from utils import mpi_master

OUTDIR = "tests/test_files/out_main_test/"


class MainTest(unittest.TestCase):
    runvars = dict(
        data_dump=os.path.join(OUTDIR, "data/fast_injection_data_dump.pickle"),
        outdir=os.path.join(OUTDIR, "result"),
        label="fast_injection_0",
    )

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
        exit_reason = analysis.analysis_runner(max_its=its, **self.runvars)

        check_master_value(exit_reason, 1)

        # Function needs to be called on all tasks because of barrier
        # in decorator
        resume_file = self.read_resume_file()

        # Only check the pickle file on master
        if MPI.COMM_WORLD.Get_rank() == 0:
            # The code runs 2 more iterations than specified
            assert resume_file.it == its + 2

    @pytest.mark.mpi
    def test_max_time(self):
        time = 0.1
        # Run analysis
        exit_reason = analysis.analysis_runner(max_run_time=time, **self.runvars)

        check_master_value(exit_reason, 2)

    @pytest.mark.mpi
    def test_resume(self):
        comm = MPI.COMM_WORLD
        # Run in full to get the reference answer
        exit_reason = analysis.analysis_runner(**self.runvars)
        # Sanity check: make sure the run actually reached the end
        check_master_value(exit_reason, 0)

        reference_result = self.read_bilby_result()

        # Reset for the resume run
        self.tearDown()
        self.setUp()

        # Run analysis for 5 iterations
        exit_reason = analysis.analysis_runner(max_its=5, **self.runvars)
        # Sanity check: make sure the run stopped because of max iterations
        check_master_value(exit_reason, 1)

        exit_reason = analysis.analysis_runner(**self.runvars)
        check_master_value(exit_reason, 0)

        resume_result = self.read_bilby_result()

        if comm.Get_rank() == 0:
            assert (
                pytest.approx(reference_result.log_evidence)
                == resume_result.log_evidence
            )

    @mpi_master
    def read_resume_file(self):
        with open(
            os.path.join(OUTDIR, "result/fast_injection_0_checkpoint_resume.pickle"),
            "rb",
        ) as f:
            resume_file = dill.load(f)
        return resume_file

    @mpi_master
    def read_bilby_result(self):
        return bilby.gw.result.CBCResult.from_json(
            os.path.join(OUTDIR, "result/fast_injection_0_result.json")
        )


@mpi_master
def check_master_value(test_value, expected_value):
    assert test_value == expected_value
