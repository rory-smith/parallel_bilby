import os.path
import pickle

import pytest
from parallel_bilby import generation, slurm
from parallel_bilby.parser import create_generation_parser, parse_generation_args
from tests.cases import GW150914Run


class GenerationTest(GW150914Run):
    def setUp(self):

        # Run the relevant bits of generation.main, with no CLI args
        cli_args = [""]
        parser = create_generation_parser()
        args = parse_generation_args(parser, cli_args=cli_args, as_namespace=True)

        # Overwrite defaults with values in test
        args.outdir = self.test_dir
        args.label = self.test_label
        args.extra_lines = "conda activate; conda source"
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

        slurm_fn = os.path.join(self.test_dir, "submit/analysis_GW150914_0.sh")
        with open(slurm_fn, "r") as file:
            slurm_script = file.read()
            self.assertTrue("mem-per-cpu" in slurm_script)
            self.assertTrue("ntasks-per-node" in slurm_script)
            self.assertTrue("nodes" in slurm_script)
            self.assertTrue("conda" in slurm_script)
