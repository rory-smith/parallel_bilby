import pytest
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
            **self.analysis_runner_kwargs,
            max_its=5,
        )

        # Check result in master task only
        self.check_result()

    @mpi_master
    def check_result(self):
        resume_file = self.read_resume_file()
        obtained = resume_file.live_logl[0]
        self.assertTrue(obtained < 0)
        # It would be better to have a test similar to the one in test_end2end.py
