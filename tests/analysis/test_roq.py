import pytest
from mpi4py import MPI
from parallel_bilby import analysis
from tests.cases import ROQRun
from tests.utils import mpi_master


class ROQTest(ROQRun):
    @pytest.mark.mpi
    def test_analysis(self):
        """
        Run a problem using ROQ for 5 iterations, and test that
        the first live point matches the reference value.
        """
        # Run analysis
        analysis.analysis_runner(
            **self.analysis_args,
            nlive=20,
            max_its=5,
        )

        # Check result in master task only
        self.check_result()

    @mpi_master
    def check_result(self):
        resume_file = self.read_resume_file()

        # The answer will vary with the number of MPI tasks
        answer = {
            2: -4.719451187977029,
            3: -0.7088613566902495,
            4: -0.7088613566902495,
            5: -4.719451187977029,
            6: -4.719451187977029,
            9: -4.719451187977029,
        }

        obtained = resume_file.live_logl[0]
        comm = MPI.COMM_WORLD
        if comm.size not in answer:
            msg = (
                f"Calculated likelihood {obtained}."
                f"Answer has not been pre-calculated for {comm.size} MPI tasks."
            )
            raise KeyError(msg)

        # The ROQ test takes too long to run to completion, so it is stopped
        # after 5 iterations, and the first live point is compared to a reference.
        # This ensures that the algorithm has not been changed. A mathematically
        # valid change to the code may change this result, so the test reference
        # value should be updated accordingly.
        if not obtained == pytest.approx(answer[comm.size], abs=1e-8):
            raise ValueError(
                f"Using {comm.size} MPI tasks, obtained {obtained}, expected {answer[comm.size]}",
            )
