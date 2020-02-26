import os
import shutil
import unittest

import mock
from src import generation

from bilby_pipe.data_generation import DataGenerationInput
from bilby_pipe.utils import DataDump

GW150914_ROOT = "examples/GW150914_IMRPhenomPv2"
GW150914_INI = f"{GW150914_ROOT}/GW150914.ini"
GW150914_PRIOR = f"{GW150914_ROOT}/GW150914.prior"
GW150914_PSD = (
    "{H1=examples/GW150914_IMRPhenomPv2/raw_data/h1_psd.txt, "
    "L1=examples/GW150914_IMRPhenomPv2/raw_data/l1_psd.txt}"
)
GW150914_TABLE = "tests/test_files/out_GW150914/.distance_marginalization_lookup.npz"


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/test_files/test_out"
        os.makedirs(self.outdir, exist_ok=True)
        self.ini = GW150914_INI

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    @staticmethod
    def get_timeseries_data():
        d = DataDump.from_pickle("tests/test_files/gwpy_data.pickle")
        timeseries = d.interferometers[0].strain_data.to_gwpy_timeseries()
        return timeseries

    @mock.patch("gwpy.timeseries.TimeSeries.fetch_open_data")
    @mock.patch("src.generation.get_cli_args")
    def get_datagen_input_object(self, get_cli_args, fetch_open_data_method):
        get_cli_args.return_value = [
            GW150914_INI,
            "--distance_marginalization_lookup_table",
            GW150914_TABLE,
        ]
        fetch_open_data_method.return_value = self.get_timeseries_data()
        args = generation.generation_parser.parse_args(args=[self.ini])
        args = generation.add_extra_args_from_bilby_pipe_namespace(args)
        args.prior_file = GW150914_PRIOR
        args.outdir = self.outdir
        args.psd_dict = GW150914_PSD
        args.submit = False
        args.distance_marginalisation = True
        args.distance_marginalization_lookup_table = GW150914_TABLE
        return DataGenerationInput(args=args, unknown_args=[])

    @mock.patch("src.generation.bilby_pipe.data_generation.DataGenerationInput")
    @mock.patch("src.generation.get_cli_args")
    @mock.patch("src.slurm.get_cli_args")
    def test_generation(self, slurm_cli, generation_cli, datagen_input):
        datagen_input.return_value = self.get_datagen_input_object()
        generation_cli.return_value = [GW150914_INI]
        slurm_cli.return_value = [GW150914_INI]
        generation.main()
        files = ["data/GW150914_data_dump.pickle", "submit/bash_GW150914.sh"]
        for f in files:
            path = os.path.join(self.outdir, f)
            self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
