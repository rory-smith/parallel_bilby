import os
import pickle
import unittest

import mock

DATA_DUMP = "tests/test_files/out_GW150914/data/GW150914_data_dump.pickle"
GW150914_ROOT = "examples/GW150914_IMRPhenomPv2"
GW150914_INI = f"{GW150914_ROOT}/GW150914.ini"
GW150914_PRIOR = f"{GW150914_ROOT}/GW150914.prior"
GW150914_PSD = (
    "{H1=examples/GW150914_IMRPhenomPv2/raw_data/h1_psd.txt, "
    "L1=examples/GW150914_IMRPhenomPv2/raw_data/l1_psd.txt}"
)
GW150914_TABLE = "tests/test_files/out_GW150914/.distance_marginalization_lookup.npz"


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/test_files/out_GW150914/"
        os.makedirs(self.outdir, exist_ok=True)
        self.data_dump = self.get_datadump()

    def get_datadump(self):
        with open(DATA_DUMP, "rb") as file:
            data_dump = pickle.load(file)
        data_dump["args"].outdir = self.outdir
        data_dump["args"].distance_marginalisation = True
        data_dump["args"].distance_marginalization_lookup_table = GW150914_TABLE
        return data_dump

    @mock.patch("pickle.load")
    @mock.patch("src.utils.get_cli_args")
    def test_analysis(self, get_args, pickle_load):
        get_args.return_value = [DATA_DUMP]
        pickle_load.return_value = self.data_dump

        with self.assertRaises(ValueError):
            # ValueError: Tried to create an MPI pool,
            # but there was only one MPI process available.
            # Need at least two.
            from src import analysis

            analysis.main()


if __name__ == "__main__":
    unittest.main()
