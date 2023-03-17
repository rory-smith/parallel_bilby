import copy
import os.path
import pickle
import re

import pytest
from parallel_bilby import generation, slurm
from parallel_bilby.parser import create_generation_parser, parse_generation_args
from tests.cases import GW150914Run
from tests.utils import dirtree


class GenerationTest(GW150914Run):
    def setUp(self):

        # Run the relevant bits of generation.main, with no CLI args
        self.cli_args = [""]
        self.parser = create_generation_parser()
        args = parse_generation_args(
            self.parser, cli_args=self.cli_args, as_namespace=True
        )

        # Overwrite defaults with values in test
        args.outdir = self.test_dir
        args.label = self.test_label
        args.extra_lines = "conda activate; conda source"
        for key, value in self.generation_args.items():
            args.__setattr__(key, value)

        self.args = args

    @pytest.mark.mpi_skip
    def test_generation(self):

        # Run generation
        args = copy.deepcopy(self.args)
        args.ntasks_per_node = 20
        args.nodes = 20
        args.n_parallel = 1
        inputs, _ = generation.generate_runner(parser=self.parser, **vars(args))
        slurm.setup_submit(inputs.data_dump_file, inputs, args, self.cli_args)

        files = [
            "GW150914_config_complete.ini",
            "data/GW150914_data_dump.pickle",
            "log_data_generation/GW150914.log",
            "submit/bash_GW150914.sh",
            "submit/analysis_GW150914_0.sh",
        ]
        self._check_generation_files(files=files, kwargs=vars(args))

    @pytest.mark.mpi_skip
    def test_generation_merge(self):

        # Run generation
        args = copy.deepcopy(self.args)
        args.n_parallel = 4
        args.mem_per_cpu = None
        args.extra_lines = "conda activate; conda source"
        args.slurm_extra_lines = "dependency=singleton partition=sstar"
        inputs, _ = generation.generate_runner(parser=self.parser, **vars(args))
        slurm.setup_submit(inputs.data_dump_file, inputs, args, self.cli_args)
        files = [
            "submit/analysis_GW150914_0.sh",
            "submit/analysis_GW150914_3.sh",
            "submit/merge_GW150914.sh",
        ]
        self._check_generation_files(files=files, kwargs=vars(args))

    @property
    def testdir_tree(self):
        return dirtree(self.test_dir)

    def _check_generation_files(self, files, kwargs):

        for f in files:

            path = os.path.join(self.test_dir, f)
            self.assertTrue(
                os.path.isfile(path), f"File {f} not found. Files:\n{self.testdir_tree}"
            )

            if "pickle" in path:
                self._check_data_dump(path=path, kwargs=kwargs)
            elif "analysis" in f and ".sh" in f:
                self._check_sbatch_file(
                    path=path,
                    kwargs=kwargs,
                    check_mem_per_cpu=True,
                    command="parallel_bilby_analysis",
                )
            elif "merge" in f and ".sh" in f:
                self._check_sbatch_file(
                    path=path,
                    kwargs=kwargs,
                    check_mem_per_cpu=False,
                    command="bilby_result -r",
                )

    def _check_data_dump(self, path, kwargs):
        with open(path, "rb") as file:
            data_dump = pickle.load(file)
            self.assertEqual(data_dump["args"].n_parallel, kwargs["n_parallel"])

    def _check_sbatch_file(
        self, path, kwargs, check_mem_per_cpu=True, command="parallel_bilby_analysis"
    ):
        with open(path, "r") as file:
            slurm_script = file.read()
            self.assertTrue(
                re.search(r"nodes=[0-9]+", slurm_script),
                "nodes not found in slurm script",
            )
            self.assertTrue(
                re.search(r"ntasks-per-node=[0-9]+", slurm_script),
                "ntasks not found in slurm script",
            )
            self.assertTrue("job-name" in slurm_script)

            if kwargs["extra_lines"] is not None:
                extra_lines_str = "\n".join(
                    [k.strip() for k in kwargs["extra_lines"].split(";")]
                )
                self.assertTrue(extra_lines_str in slurm_script)

            if kwargs["slurm_extra_lines"] is not None:
                slurm_extra_lines_str = "\n".join(
                    [
                        "#SBATCH --" + line
                        for line in kwargs["slurm_extra_lines"].split(" ")
                    ]
                )
                self.assertTrue(slurm_extra_lines_str in slurm_script)

            if check_mem_per_cpu:  # only check in analysis scripts
                if kwargs["mem_per_cpu"] is not None:
                    self.assertTrue("mem-per-cpu" in slurm_script)
                else:
                    self.assertFalse("mem-per-cpu" in slurm_script)

            self.assertTrue(
                command in slurm_script.splitlines()[-1].strip(),
                "Command not found in slurm script",
            )
