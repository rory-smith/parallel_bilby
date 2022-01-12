import os
import shutil
import unittest

from parallel_bilby import generation
from parallel_bilby.analysis import analysis_run


class AnalysisRunTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/test_files/out_analysis_run_test/"

        # Use same ini file as fast e2e test
        generation.generate_runner(
            ["tests/test_files/fast_test.ini", "--outdir", self.outdir]
        )

        self.run = analysis_run.AnalysisRun(
            data_dump=os.path.join(self.outdir, "data/fast_injection_data_dump.pickle"),
            outdir=self.outdir,
        )

    def tearDown(self):
        shutil.rmtree(self.outdir)

    def test_prior_transform_function(self):
        # Fast test only has one parameter: the chirp mass
        assert self.run.prior_transform_function([0])[0] == 25
        assert self.run.prior_transform_function([0.5])[0] == 28
        assert self.run.prior_transform_function([1])[0] == 31
