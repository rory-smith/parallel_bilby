import os
import shutil
from argparse import Namespace
from unittest import TestCase

import mock

from src import generation

from . import get_timeseries, ini, psd_strain_file, strain_file


class TestGeneration(TestCase):
    def setUp(self):
        self.ini = ini
        self.strain_file = strain_file
        self.psd_strain_file = psd_strain_file
        self.maxDiff = 0

    def tearDown(self):
        if os.path.isdir("outdir"):
            shutil.rmtree("outdir")

    def test_arg_parser(self):
        with self.assertRaises(SystemExit):
            generation.get_args()

    @mock.patch("src.generation.get_args")
    @mock.patch("src.generation.get_cli_args")
    @mock.patch("bilby_pipe.data_generation.DataGenerationInput._gwpy_get")
    @mock.patch("bilby_pipe.data_generation.DataGenerationInput._is_gwpy_data_good")
    def test_get_data(self, is_data_good, get_data_method, cli_args, get_args):
        h1_strain, h1_psd = get_timeseries()
        get_data_method.side_effect = [h1_strain, h1_psd]
        is_data_good.return_value = True
        get_args.return_value = Namespace(
            outdir="outdir",
            label="test",
            trigger_time=1126259462.4,
            duration=4,
            psd_dict={"H1": "tests/test_files/raw_data/h1_psd.txt"},
            prior_file="tests/test_files/test.prior",
            waveform_approximant="IMRPhenomPv2",
            distance_marginalization=False,
            phase_marginalization=True,
            time_marginalization=True,
            data_dict=None,
            binary_neutron_star=False,
            deltaT=0.2,
            sampling_frequency=4096,
            minimum_frequency=20,
            maximum_frequency=2048,
            calibration_model=None,
            reference_frequency=20,
            distance_marginalization_lookup_table=None,
        )
        cli_args.return_value = ["tests/test_files/test.ini"]
        generation.main()
        self.assertTrue(os.path.isfile("outdir/data/H1_strain.hdf5"))
        generation.main()  # testing the removal of cached strain data
