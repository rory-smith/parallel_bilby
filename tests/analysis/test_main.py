import os.path
import shutil
import unittest

import dill
import pytest
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
        try:
            analysis.analysis_runner(
                data_dump=os.path.join(OUTDIR, "data/fast_injection_data_dump.pickle"),
                outdir=os.path.join(OUTDIR, "result"),
                label="fast_injection_0",
                max_its=its,
            )
        # sys.exit(0) is called when max_its is reached
        except SystemExit as e:
            assert e.code == 0

        with open(
            os.path.join(OUTDIR, "result/fast_injection_0_checkpoint_resume.pickle"),
            "rb",
        ) as f:
            resume_file = dill.load(f)

        # The code runs 2 more iterations than specified
        assert resume_file.it == its + 2
