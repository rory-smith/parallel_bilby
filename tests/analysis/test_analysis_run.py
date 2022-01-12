import os
import shutil
import unittest

import numpy as np
import pytest
from parallel_bilby import generation
from parallel_bilby.analysis import analysis_run
from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool
from utils import mpi_master


class AnalysisRunTest(unittest.TestCase):
    @mpi_master
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

        self.min_chirp_mass = self.run.priors["chirp_mass"].minimum
        self.max_chirp_mass = self.run.priors["chirp_mass"].maximum

    @mpi_master
    def tearDown(self):
        shutil.rmtree(self.outdir)

    def test_prior_transform_function(self):
        # Fast test only has one parameter: the chirp mass
        assert self.run.prior_transform_function([0])[0] == self.min_chirp_mass
        assert self.run.prior_transform_function([0.5])[0] == 0.5 * (
            self.min_chirp_mass + self.max_chirp_mass
        )
        assert self.run.prior_transform_function([1])[0] == self.max_chirp_mass

    def test_log_likelihood_function(self):
        v_array = self.run.prior_transform_function([0.5])
        assert pytest.approx(878.8714803944144) == self.run.log_likelihood_function(
            v_array
        )

        v_array = self.run.prior_transform_function([100])
        assert np.nan_to_num(-np.inf) == self.run.log_likelihood_function(v_array)

        self.run.zero_likelihood_mode = True
        assert 0 == self.run.log_likelihood_function(v_array)

    def test_log_prior_function(self):
        for i in np.linspace(0, 1, 4):
            v_array = self.run.prior_transform_function([i])
            assert pytest.approx(
                np.log(1 / (self.max_chirp_mass - self.min_chirp_mass))
            ) == self.run.log_prior_function(v_array)

    @pytest.mark.mpi
    def test_get_initial_points_from_prior(self):

        # Create a test pool
        with MPIPool() as pool:
            if pool.is_master():
                unit, theta, loglike = self.run.get_initial_points_from_prior(pool)

                # Check arrays have correct length
                assert len(unit) == self.run.nlive
                assert len(theta) == self.run.nlive
                assert len(loglike) == self.run.nlive

                for i in range(self.run.nlive):
                    # Check point is valid
                    self._check_point_validity(unit[i], theta[i], loglike[i])

                    # Check point is unique
                    assert 1 == np.sum(np.isclose(unit[i], unit))
                    assert 1 == np.sum(np.isclose(theta[i], theta))
                    assert 1 == np.sum(np.isclose(loglike[i], loglike))

    def test_get_initial_point_from_prior(self):
        args = (
            self.run.prior_transform_function,
            self.run.log_prior_function,
            self.run.log_likelihood_function,
            len(self.run.sampling_keys),
            True,
            np.random.Generator(np.random.PCG64(1)),
        )

        unit, theta, loglike = self.run.get_initial_point_from_prior(args)
        self._check_point_validity(unit[0], theta[0], loglike)

    def _check_point_validity(self, unit, theta, loglike):
        # Check that the values are sensible
        assert 0 <= unit <= 1
        assert self.min_chirp_mass <= theta <= self.max_chirp_mass

        v_array = self.run.prior_transform_function([unit])
        assert pytest.approx(loglike) == self.run.log_likelihood_function(v_array)

    # def test_get_nested_sampler(self):
    #     # Create a test pool
    #     pool = multiprocessing.Pool(4)

    #     self.run.get_nested_sampler()
