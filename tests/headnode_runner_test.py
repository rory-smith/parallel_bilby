import multiprocessing
import os
import shutil
import subprocess
import unittest
from multiprocessing import Pool

import pytest
from mpi4py import MPI

INI = "tests/test_files/injection.ini"


class HeadnodeRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.original_dir = os.getcwd()
        self.ini = INI
        self.ini_dir = os.path.dirname(INI)
        self.outdir = "tests/test_files/outdir_test_headnode"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        os.chdir(self.original_dir)
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_headnode_run(self):
        pool = Pool(processes=4)
        comm = MPI.COMM_WORLD
        self.assertTrue(comm.size >= 2, comm.size)
        os.chdir(self.ini_dir)
        log = subprocess.run(["parallel_bilby_generation", "injection.ini"])
        # test
        log = subprocess.run(["parallel_bilby_generation", "injection.ini"])

        "mpirun -n 2 parallel_bilby_analysis outdir_test_headnode/data/test_headnode_data_dump.pickle --maxmcmc 10000 --nact 2 --label test_headnode_0 --outdir /Users/avaj0001/Documents/projects/parallel_bilby/tests/test_files/outdir_test_headnode/result --sampling-seed 1234"


if __name__ == "__main__":
    unittest.main()
