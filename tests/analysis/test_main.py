import pytest
from mpi4py import MPI
from parallel_bilby import analysis
from tests.cases import FastRun
from tests.utils import mpi_master


class MainTest(FastRun):
    @pytest.mark.mpi
    def test_max_its(self):
        """
        Test that the code halts and writes a checkpoint correctly once the
        maximum number of iterations has been reached.
        """
        its = 5
        # Run analysis
        exit_reason = analysis.analysis_runner(max_its=its, **self.analysis_args)

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
        """
        Test that the code halts and writes a checkpoint correctly once the
        maximum wall time has been reached.
        """
        time = 0.1
        # Run analysis
        exit_reason = analysis.analysis_runner(max_run_time=time, **self.analysis_args)

        check_master_value(exit_reason, 2)

    @pytest.mark.mpi
    def test_resume(self):
        """
        Test that the code writes and reads from a checkpoint, and produces the
        same result as if it had run from beginning to end without stopping.
        """
        comm = MPI.COMM_WORLD
        # Run in full to get the reference answer
        exit_reason = analysis.analysis_runner(**self.analysis_args)
        # Sanity check: make sure the run actually reached the end
        check_master_value(exit_reason, 0)

        reference_result = self.read_bilby_result()

        # Reset for the resume run
        self.tearDown()
        self.setUp()

        # Run analysis for 5 iterations
        exit_reason = analysis.analysis_runner(max_its=5, **self.analysis_args)
        # Sanity check: make sure the run stopped because of max iterations
        check_master_value(exit_reason, 1)

        exit_reason = analysis.analysis_runner(**self.analysis_args)
        check_master_value(exit_reason, 0)

        resume_result = self.read_bilby_result()

        if comm.Get_rank() == 0:
            assert (
                pytest.approx(reference_result.log_evidence)
                == resume_result.log_evidence
            )


@mpi_master
def check_master_value(test_value, expected_value):
    """
    Function with the @mpi_master decorator applied so that
    the value is only checked on the master task
    """
    assert test_value == expected_value
