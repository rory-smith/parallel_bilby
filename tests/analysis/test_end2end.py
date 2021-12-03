import os
import pickle
import unittest
from mpi4py import MPI
import shutil
import bilby

from parallel_bilby import analysis, generation

def mpi_master(func):
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            f = func(*args, **kwargs)
        else:
            f = None
        comm.Barrier()
        return f
    return wrapper

class AnalysisTest(unittest.TestCase):
    @mpi_master
    def setUp(self):
        self.outdir = "tests/test_files/out_fast/"
        generation.generate_runner(['tests/test_files/fast_test.ini', '--outdir', self.outdir])

    @mpi_master
    def tearDown(self):
        shutil.rmtree(self.outdir)

    def test_analysis(self):
        # Run analysis
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

        # Check result in master task only
        self.check_result()

    @mpi_master
    def check_result(self):
        # Read file and check result
        b = bilby.gw.result.CBCResult.from_json(os.path.join(self.outdir, 'result/fast_injection_0_result.json'))
        result = b.log_evidence
        print(result)
