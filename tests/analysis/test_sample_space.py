import numpy as np
import pandas as pd
import pytest
from deepdiff import DeepDiff
from parallel_bilby.analysis import analysis_run
from parallel_bilby.analysis.sample_space import fill_sample
from tests.cases import FastRun


class SampleSpaceTest(FastRun):
    def setUp(self):
        super().setUp()
        self.run = analysis_run.AnalysisRun(**self.analysis_args)

    @pytest.mark.mpi_skip
    def test_fill_sample(self):
        ii = 0
        row = pd.Series(
            dict(
                chirp_mass=28.005628,
                log_likelihood=878.837994,
                log_prior=0,
                mass_ratio=1.0,
                a_1=0.6,
                a_2=0.6,
                tilt_1=0.0,
                tilt_2=0.0,
                phi_12=0.0,
                phi_jl=0.0,
                luminosity_distance=800.0,
                dec=0.1,
                ra=0.1,
                theta_jn=0.1,
                psi=0.1,
                phase=0.1,
                geocent_time=0.0,
            )
        )

        sample = fill_sample((ii, row, self.run.likelihood))
        reference_sample = {
            "mass_ratio": 1.0,
            "a_1": 0.6,
            "a_2": 0.6,
            "tilt_1": 0.0,
            "tilt_2": 0.0,
            "phi_12": 0.0,
            "phi_jl": 0.0,
            "luminosity_distance": 800,
            "dec": 0.1,
            "ra": 0.1,
            "theta_jn": 0.1,
            "psi": 0.1,
            "phase": 0.1,
            "geocent_time": 0,
            "chirp_mass": 28.005627999999994,
            "log_likelihood": 878.837994,
            "log_prior": 0.0,
            "H1_matched_filter_snr": (41.924647195266836 - 0.2023449501540067j),
            "H1_optimal_snr": 41.93240615236018,
            "reference_frequency": 20.0,
            "waveform_approximant": "IMRPhenomPv2",
            "minimum_frequency": 20.0,
            "maximum_frequency": 2048.0,
            "catch_waveform_errors": False,
            "pn_spin_order": -1,
            "pn_tidal_order": -1,
            "pn_phase_order": -1,
            "pn_amplitude_order": 0,
            "mode_array": None,
            "total_mass": 64.3400376285178,
            "mass_1": 32.1700188142589,
            "mass_2": 32.1700188142589,
            "symmetric_mass_ratio": 0.25,
            "iota": np.array(0.1),
            "spin_1x": np.array(0),
            "spin_1y": np.array(0),
            "spin_1z": np.array(0.6),
            "spin_2x": np.array(0),
            "spin_2y": np.array(0),
            "spin_2z": np.array(0.6),
            "phi_1": 0.0,
            "phi_2": 0.0,
            "chi_eff": 0.6,
            "chi_1_in_plane": 0.0,
            "chi_2_in_plane": 0.0,
            "chi_p": 0.0,
            "cos_tilt_1": 1.0,
            "cos_tilt_2": 1.0,
            "redshift": np.array(0.161815778428732),
            "comoving_distance": 688.577324965461,
            "mass_1_source": 27.68943184587011,
            "mass_2_source": 27.68943184587011,
            "chirp_mass_source": 24.10505049077186,
            "total_mass_source": 55.37886369174022,
        }

        assert {} == DeepDiff(
            sample,
            reference_sample,
            number_format_notation="e",
            significant_digits=6,
            ignore_numeric_type_changes=True,
            exclude_types=[np.ndarray],
        )

        # Numpy comparison for 0-d arrays does not work with DeepDiff
        # Handle them separately.
        for key in reference_sample:
            if type(reference_sample[key]) == np.ndarray:
                assert reference_sample[key] == pytest.approx(sample[key], abs=1e-8)
