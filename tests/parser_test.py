import unittest
from argparse import Namespace

import numpy as np
from parallel_bilby.parser import create_analysis_parser, create_generation_parser

GW150914_INI = "examples/GW150914_IMRPhenomPv2/GW150914.ini"
TEST_INI = "tests/test_files/test_bilby_pipe_args.ini"
DATA_DUMP = "data_dump.pickle"


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    @staticmethod
    def get_generation_expected_args():
        return Namespace(
            catch_waveform_errors=False,
            bilby_zero_likelihood_mode=False,
            calibration_model=None,
            channel_dict="{H1:DCS-CALIB_STRAIN_C02, L1:DCS-CALIB_STRAIN_C02}",
            deltaT=0.2,
            detectors=["H1", "L1"],
            distance_marginalization=True,
            dlogz=0.1,
            do_not_save_bounds_in_resume=False,
            duration=4.0,
            dynesty_bound="multi",
            dynesty_sample="rwalk",
            facc=0.5,
            frequency_domain_source_model="lal_binary_black_hole",
            gaussian_noise=False,
            ini="examples/GW150914_IMRPhenomPv2/GW150914.ini",
            injection=False,
            injection_file=None,
            injection_numbers=None,
            injection_dict=None,
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
            n_check_point=100,
            n_effective=np.inf,
            n_simulation=0,
            n_parallel=4,
            nact=5,
            nlive=1000,
            nodes=10,
            ntasks_per_node=16,
            outdir="outdir",
            phase_marginalization=True,
            post_trigger_duration=2.0,
            prior_file=None,
            psd_dict="{H1=psd_data/h1_psd.txt, L1=psd_data/l1_psd.txt}",
            time="24:00:00",
            time_marginalization=True,
            trigger_time=1126259462.4,
            tukey_roll_off=0.4,
            vol_check=8,
            vol_dec=0.5,
            walks=100,
            waveform_approximant="IMRPhenomPv2",
            waveform_generator="bilby.gw.waveform_generator.WaveformGenerator",
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
        )

    def test_generation_parser(self):
        parser = create_generation_parser()
        args = parser.parse_args(args=[GW150914_INI])
        expected_args = self.get_generation_expected_args()

        args_dict = vars(args)
        expected_args_dict = vars(expected_args)

        for arg_key in expected_args_dict.keys():
            self.assertEqual(args_dict[arg_key], expected_args_dict[arg_key], arg_key)

    def test_generation_parser_for_bilby_pipe_args(self):
        parser = create_generation_parser()
        args = parser.parse_args(args=[TEST_INI])
        self.assertEqual(args.n_parallel, 4)

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
            n_check_point=100,
            n_effective=np.inf,
            check_point_deltaT=600,
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
