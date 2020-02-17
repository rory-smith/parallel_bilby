import os
import shutil
from unittest import TestCase

import mock

from src.data_retrieval import main

from . import get_timeseries, ini, psd_strain_file, strain_file


class TestDataRetrieval(TestCase):
    def setUp(self):
        self.ini = ini
        self.strain_file = strain_file
        self.psd_strain_file = psd_strain_file
        self.maxDiff = 0

    def tearDown(self):
        if os.path.isdir("outdir"):
            shutil.rmtree("outdir")

    @mock.patch("src.data_retrieval.get_cli_args")
    @mock.patch("bilby_pipe.data_generation.DataGenerationInput._gwpy_get")
    @mock.patch("bilby_pipe.data_generation.DataGenerationInput._is_gwpy_data_good")
    def test_get_data(self, is_data_good, get_data_method, cli_args):
        h1_strain, h1_psd = get_timeseries()
        l1_strain, l1_psd = get_timeseries()
        v1_strain, v1_psd = get_timeseries()

        get_data_method.side_effect = [
            h1_strain,
            h1_psd,
            l1_strain,
            l1_psd,
            v1_strain,
            v1_psd,
        ]
        is_data_good.return_value = True
        cli_args.return_value = [self.ini]
        main()
