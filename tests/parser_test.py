import unittest
from argparse import Namespace

import numpy as np
from parallel_bilby.parser import create_analysis_parser, create_generation_parser

GW150914_INI = "examples/GW150914_IMRPhenomPv2/GW150914.ini"
TEST_INI = "tests/test_files/test.ini"
DATA_DUMP = "data_dump.pickle"


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_generation_parser(self):
        parser = create_generation_parser()
        args = parser.parse_args(args=[GW150914_INI])
        expected_args = Namespace(
            bilby_zero_likelihood_mode=False,
            calibration_model=None,
            channel_dict="{H1:DCS-CALIB_STRAIN_C02, L1:DCS-CALIB_STRAIN_C02}",
            clean=False,
            coherence_test=False,
            convert_to_flat_in_component_mass=False,
            create_plots=False,
            create_summary=False,
            data_dict=None,
            data_format=None,
            default_prior="BBHPriorDict",
            deltaT=0.2,
            detectors=["H1", "L1"],
            distance_marginalization=True,
            distance_marginalization_lookup_table=None,
            dlogz=0.1,
            do_not_save_bounds_in_resume=False,
            duration=4.0,
            dynesty_bound="multi",
            dynesty_sample="rwalk",
            enlarge=1.5,
            existing_dir=None,
            extra_lines=None,
            facc=0.5,
            frequency_domain_source_model="lal_binary_black_hole",
            gaussian_noise=False,
            generation_seed=None,
            gps_file=None,
            gps_tuple=None,
            ignore_gwpy_data_quality_check=True,
            ini="examples/GW150914_IMRPhenomPv2/GW150914.ini",
            injection=False,
            injection_file=None,
            injection_numbers=None,
            injection_waveform_approximant=None,
            jitter_time=True,
            label="GW150914",
            likelihood_type="GravitationalWaveTransient",
            log_directory=None,
            maximum_frequency=None,
            maxmcmc=5000,
            mem_per_cpu="1000",
            min_eff=10,
            minimum_frequency="20",
            n_check_point=100000,
            n_effective=np.inf,
            n_simulation=0,
            n_parallel=1,
            nact=5,
            nlive=1000,
            no_plot=False,
            nodes=1,
            ntasks_per_node=1,
            outdir="outdir",
            periodic_restart_time=43200,
            phase_marginalization=True,
            post_trigger_duration=2.0,
            prior_file="GW150914.prior",
            psd_dict="{H1=raw_data/h1_psd.txt, L1=raw_data/l1_psd.txt}",
            psd_fractional_overlap=0.5,
            psd_length=32,
            psd_maximum_duration=1024,
            psd_method="median",
            psd_start_time=None,
            reference_frequency=20,
            roq_folder=None,
            roq_scale_factor=1,
            sampling_frequency=4096,
            sampling_seed=1234,
            spline_calibration_amplitude_uncertainty_dict=None,
            spline_calibration_envelope_dict=None,
            spline_calibration_nodes=5,
            spline_calibration_phase_uncertainty_dict=None,
            submit=False,
            summarypages_arguments=None,
            time="24:00:00",
            time_marginalization=True,
            timeslide_file=None,
            trigger_time=1126259462.4,
            tukey_roll_off=0.4,
            verbose=False,
            vol_check=8,
            vol_dec=0.5,
            walks=100,
            waveform_approximant="IMRPhenomPv2",
            webdir=None,
            zero_noise=False,
            burn_in_nact=50.0,
            check_point_deltaT=600,
            frac_threshold=0.01,
            max_iterations=100000,
            min_tau=30,
            ncheck=500,
            nfrac=5,
            nsamples=10000,
            ntemps=20,
            nwalkers=100,
            safety=1.0,
            sampler="dynesty",
            thin_by_nact=1.0,
            Tmax=10000,
            autocorr_c=5.0,
            autocorr_tol=50.0,
            adapt=False,
            slurm_extra_lines=None,
        )
        self.assertDictEqual(vars(args), vars(expected_args))

    def test_analysis_parser(self):
        parser = create_analysis_parser()
        args = parser.parse_args(args=[DATA_DUMP])
        expected_args = Namespace(
            bilby_zero_likelihood_mode=False,
            clean=False,
            data_dump=DATA_DUMP,
            dlogz=0.1,
            do_not_save_bounds_in_resume=False,
            dynesty_bound="multi",
            dynesty_sample="rwalk",
            enlarge=1.5,
            facc=0.5,
            label=None,
            maxmcmc=5000,
            min_eff=10,
            n_check_point=100000,
            n_effective=np.inf,
            nact=5,
            nlive=1000,
            no_plot=False,
            outdir=None,
            sampling_seed=1234,
            vol_check=8,
            vol_dec=0.5,
            walks=100,
        )
        self.assertDictEqual(vars(args), vars(expected_args))


if __name__ == "__main__":
    unittest.main()
