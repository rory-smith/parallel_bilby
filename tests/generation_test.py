import os
import pickle
import shutil
import unittest

import mock
import numpy as np
from bilby.core.prior import Constraint, Cosine, PowerLaw, Sine, Uniform
from bilby_pipe.data_generation import DataGenerationInput
from bilby_pipe.utils import DataDump
from parallel_bilby import generation

GW150914_ROOT = "examples/GW150914_IMRPhenomPv2"
GW150914_INI = f"{GW150914_ROOT}/GW150914.ini"
GW150914_PRIOR = f"{GW150914_ROOT}/GW150914.prior"
GW150914_PSD = (
    "{H1=examples/GW150914_IMRPhenomPv2/psd_data/h1_psd.txt, "
    "L1=examples/GW150914_IMRPhenomPv2/psd_data/l1_psd.txt}"
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
    @mock.patch("parallel_bilby.generation.get_cli_args")
    def get_datagen_input_object(self, get_cli_args, fetch_open_data_method):
        get_cli_args.return_value = [
            GW150914_INI,
            "--distance_marginalization_lookup_table",
            GW150914_TABLE,
        ]
        fetch_open_data_method.return_value = self.get_timeseries_data()
        args = generation.generation_parser.parse_args(args=[self.ini])
        args = generation.add_extra_args_from_bilby_pipe_namespace(args)
        args.prior_dict = dict(
            mass_ratio=Uniform(name="mass_ratio", minimum=0.125, maximum=1),
            chirp_mass=Uniform(name="chirp_mass", minimum=25, maximum=31),
            mass_1=Constraint(name="mass_1", minimum=10, maximum=80),
            mass_2=Constraint(name="mass_2", minimum=10, maximum=80),
            a_1=Uniform(name="a_1", minimum=0, maximum=0.99),
            a_2=Uniform(name="a_2", minimum=0, maximum=0.99),
            tilt_1=Sine(name="tilt_1"),
            tilt_2=Sine(name="tilt_2"),
            phi_12=Uniform(
                name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            phi_jl=Uniform(
                name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
            luminosity_distance=PowerLaw(
                alpha=2, name="luminosity_distance", minimum=50, maximum=2000
            ),
            dec=Cosine(name="dec"),
            ra=Uniform(name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"),
            theta_jn=Sine(name="theta_jn"),
            psi=Uniform(name="psi", minimum=0, maximum=np.pi, boundary="periodic"),
            phase=Uniform(
                name="phase", minimum=0, maximum=2 * np.pi, boundary="periodic"
            ),
        )
        args.outdir = self.outdir
        args.psd_dict = GW150914_PSD
        args.submit = False
        args.distance_marginalisation = True
        args.distance_marginalization_lookup_table = GW150914_TABLE
        return DataGenerationInput(args=args, unknown_args=[])

    @mock.patch(
        "parallel_bilby.generation.bilby_pipe.data_generation.DataGenerationInput"
    )
    @mock.patch("parallel_bilby.generation.get_cli_args")
    @mock.patch("parallel_bilby.slurm.get_cli_args")
    def test_generation(self, slurm_cli, generation_cli, datagen_input):
        datagen_input.return_value = self.get_datagen_input_object()
        generation_cli.return_value = [GW150914_INI]
        slurm_cli.return_value = [GW150914_INI]
        generation.main()
        files = [
            "data/GW150914_data_dump.pickle",
            "submit/bash_GW150914.sh",
            "submit/analysis_GW150914_0.sh",
            "submit/analysis_GW150914_1.sh",
            "submit/analysis_GW150914_2.sh",
            "submit/analysis_GW150914_3.sh",
        ]
        for f in files:
            path = os.path.join(self.outdir, f)
            self.assertTrue(
                os.path.isfile(path), f"After generation the file {f} not found"
            )
            if "pickle" in path:
                with open(path, "rb") as file:
                    data_dump = pickle.load(file)
                    self.assertTrue(data_dump["args"].n_parallel, 4)


if __name__ == "__main__":
    unittest.main()
