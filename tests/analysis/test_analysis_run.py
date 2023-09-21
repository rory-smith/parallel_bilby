import numpy as np
import pytest
from bilby.core.sampler.dynesty_utils import LivePointSampler
from parallel_bilby.analysis import analysis_run
from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool
from tests.cases import FastRun


class AnalysisRunTest(FastRun):
    def setUp(self):
        super().setUp()
        self.run = analysis_run.AnalysisRun(**self.analysis_args)
        self.min_chirp_mass = self.run.priors["chirp_mass"].minimum
        self.max_chirp_mass = self.run.priors["chirp_mass"].maximum

    @pytest.mark.mpi_skip
    def test_prior_transform_function(self):
        # Fast test only has one parameter: the chirp mass
        assert self.run.prior_transform_function([0])[0] == self.min_chirp_mass
        assert self.run.prior_transform_function([0.5])[0] == 0.5 * (
            self.min_chirp_mass + self.max_chirp_mass
        )
        assert self.run.prior_transform_function([1])[0] == self.max_chirp_mass

    @pytest.mark.mpi_skip
    def test_log_likelihood_function(self):
        v_array = self.run.prior_transform_function([0.5])
        calc_lnl = self.run.log_likelihood_function(v_array)
        assert pytest.approx(-70, rel=1) == calc_lnl

        v_array = self.run.prior_transform_function([100])
        assert np.nan_to_num(-np.inf) == self.run.log_likelihood_function(v_array)

        self.run.zero_likelihood_mode = True
        assert 0 == self.run.log_likelihood_function(v_array)

    @pytest.mark.mpi_skip
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
                unit, theta, loglike, _ = self.run.get_initial_points_from_prior(pool)

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

    @pytest.mark.mpi_skip
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

    @pytest.mark.mpi_skip
    def _check_point_validity(self, unit: float, theta: float, loglike: float):
        # Check that the values are sensible
        assert 0 <= unit <= 1
        assert self.min_chirp_mass <= theta <= self.max_chirp_mass

        v_array = self.run.prior_transform_function([unit])
        test_loglike = self.run.log_likelihood_function(v_array)
        self.assertAlmostEqual(loglike, test_loglike, places=5)

    @pytest.mark.mpi
    def test_get_nested_sampler(self):
        # Create a test pool
        with MPIPool() as pool:
            if pool.is_master():
                live_points = self.run.get_initial_points_from_prior(pool)
                sampler = self.run.get_nested_sampler(live_points, pool, pool.size)

                assert isinstance(sampler, LivePointSampler)
