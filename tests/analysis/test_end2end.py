import numpy as np
import pytest
from mpi4py import MPI
from parallel_bilby import analysis
from tests.cases import FastRun
from tests.utils import mpi_master


class End2EndTest(FastRun):
    @pytest.mark.mpi
    def test_analysis(self):
        # Run analysis
        analysis.analysis_runner(**self.analysis_args)

        # Check result in master task only
        self.check_result()

    @mpi_master
    def check_result(self):
        # Read file and check result
        b = self.read_bilby_result()

        # The answer will vary with the number of MPI tasks
        answer = {
            2: -4.323050871371379,
            3: -4.858974047162405,
            4: -5.658712591755034,
            5: -4.263809701732271,
            6: -4.855465339795728,
            7: -4.78252357558722,
            8: -5.4694055472964465,
            9: -4.21178229675354,
        }

        values = list(answer.values())
        avg = np.mean(values)
        std = np.std(values)

        comm = MPI.COMM_WORLD
        if comm.size not in answer:
            msg = f"""
                Calculated evidence = {b.log_evidence}
                Answer has not been pre-calculated for {comm.size} MPI tasks
                """
            diff = abs(b.log_evidence - avg)
            if diff > std:
                msg += "Calculated evidence lies outside of 1 standard deviation of known reference values"

            raise KeyError(msg)

        assert b.log_evidence == pytest.approx(answer[comm.size], abs=1e-12)
