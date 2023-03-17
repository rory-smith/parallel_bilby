import numpy as np
import pytest
from mpi4py import MPI
from parallel_bilby import analysis
from tests.cases import FastRun
from tests.utils import logger, mpi_master


class End2EndTest(FastRun):
    @pytest.mark.mpi
    def test_analysis(self):
        """
        Run a short problem in full and check that the answer has not changed.
        """
        # Run analysis
        analysis.analysis_runner(**self.analysis_runner_kwargs)

        # Check result in master task only
        self.check_result()

    @mpi_master
    def check_result(self):
        # Read file and check result
        b = self.read_bilby_result()

        # The answer will vary with the number of MPI tasks
        answer = {
            2: -3.810774372114338,
            3: -4.858974047162405,
            4: -5.658712591755034,
            5: -4.263809701732271,
            6: -3.855465339795728,
            7: -4.78252357558722,
            8: -5.4694055472964465,
            9: -5.21178229675354,
        }

        values = list(answer.values())
        avg = np.mean(values)
        std = np.std(values) * 2

        comm = MPI.COMM_WORLD
        if comm.size in answer:
            exact_match = b.log_evidence == pytest.approx(answer[comm.size], abs=1e-12)
            if not exact_match:
                logger.info(
                    f"Using {comm.size} MPI tasks, calculated evidence = {b.log_evidence}, expected {answer[comm.size]}"
                )

        self.assertTrue(
            avg - std < b.log_evidence < avg + std,
            f"Calculated evidence {b.log_evidence:.2f} lies outside of 2stdpyt "
            f"of known reference values ({avg:.2f} +/- {std:.2f})",
        )
