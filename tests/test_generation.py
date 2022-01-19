import os.path
import pickle

import pytest
from parallel_bilby import generation, slurm
from parallel_bilby.parser import create_generation_parser
from tests.cases import GW150914Run


class GenerationTest(GW150914Run):
    def setUp(self):

        # Run the relevant bits of generation.main, with no CLI args
        parser = create_generation_parser()
        cli_args = [""]
        args = parser.parse_args(args=cli_args)
        args = generation.add_extra_args_from_bilby_pipe_namespace(cli_args, args)

        # Overwrite defaults with values in test
        args.outdir = self.test_dir
        args.label = self.test_label
        for key, value in self.generation_args.items():
            args.__setattr__(key, value)

        # Run generation
        inputs, _ = generation.generate_runner(parser=parser, **vars(args))

        # Write slurm script
        slurm.setup_submit(inputs.data_dump_file, inputs, args, cli_args)

    @pytest.mark.mpi_skip
    def test_generation(self):
        files = [
            "GW150914_config_complete.ini",
            "data/GW150914_data_dump.pickle",
            "log_data_generation/GW150914.log",
            "submit/bash_GW150914.sh",
            "submit/analysis_GW150914_0.sh",
        ]
        for f in files:
            path = os.path.join(self.test_dir, f)
            self.assertTrue(
                os.path.isfile(path), f"After generation the file {f} not found. Files"
            )
            if "pickle" in path:
                with open(path, "rb") as file:
                    data_dump = pickle.load(file)
                    self.assertTrue(data_dump["args"].n_parallel, 4)
