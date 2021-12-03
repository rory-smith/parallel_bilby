import os
import pickle
import unittest
from mpi4py import MPI
import mock

from parallel_bilby import analysis, generation

def mpi_master(func):
    def wrapper(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

class AnalysisTest(unittest.TestCase):
    @mpi_master
    def setUp(self):
        self.outdir = "tests/test_files/out_fast/"
        generation.generate_runner(['tests/test_files/fast_test.ini', '--outdir', self.outdir])

    def test_analysis(self):
        analysis.analysis_runner([
            'tests/test_files/out_fast/data/fast_injection_data_dump.pickle',
            '--nlive', '5',
            '--dlogz', '10.0',
            '--nact', '1',
            '--n-check-point', '10000',
            '--label', 'fast_injection_0',
            '--outdir', 'tests/test_files/out_fast/result',
            '--sampling-seed', '0',
        ])
