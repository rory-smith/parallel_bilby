import shutil
import unittest

import dill
import pytest
from mpi4py import MPI
from parallel_bilby import analysis, generation
from utils import mpi_master


class ROQTest(unittest.TestCase):
    @mpi_master
    def setUp(self):
        self.outdir = "tests/test_files/roq/out_roq/"
        generation.generate_runner(
            ["tests/test_files/roq/pbilby_roq_4s_test.ini", "--outdir", self.outdir]
        )

    @mpi_master
    def tearDown(self):
        pass
        shutil.rmtree(self.outdir)

    @pytest.mark.mpi
    def test_analysis(self):
        # Run analysis
        analysis.analysis_runner(
            data_dump="tests/test_files/roq/out_roq/data/roq_4s_test_data_dump.pickle",
            outdir="tests/test_files/roq/out_roq/result",
            label="roq_4s_test",
            nlive=20,
            max_its=5,
        )

        # Check result in master task only
        self.check_result()

    @mpi_master
    def check_result(self):
        with open(
            "tests/test_files/roq/out_roq/result/roq_4s_test_checkpoint_resume.pickle",
            "rb",
        ) as f:
            resume_file = dill.load(f)

        print(resume_file.live_logl)

        # The answer will vary with the number of MPI tasks
        answer = {
            2: -9.33063626e-01,
            3: -0.7084685344270838,
            4: -0.7084685344270838,
            5: -4.716500907386944,
            6: -4.716500907386944,
            7: -4.716500907386944,
            8: -4.716500907386944,
            9: -4.716500907386944,
        }

        comm = MPI.COMM_WORLD
        if comm.size not in answer:
            msg = f"""
                Answer has not been pre-calculated for {comm.size} MPI tasks
                """
            raise KeyError(msg)

        # The ROQ test takes too long to run to completion, so it is stopped
        # after 5 iterations, and the first live point is compared to a reference.
        # This ensures that the algorithm has not been changed. A mathematically
        # valid change to the code may change this result, so the test reference
        # value should be updated accordingly.
        assert resume_file.live_logl[0] == pytest.approx(answer[comm.size])
