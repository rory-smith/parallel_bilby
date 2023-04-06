import os
from argparse import Namespace
from os.path import abspath

import jinja2
from parallel_bilby.parser import create_analysis_parser

DIR = os.path.dirname(__file__)
TEMPLATE_SLURM = "template_slurm.sh"


def load_template(template_file: str):
    template_loader = jinja2.FileSystemLoader(searchpath=DIR)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_file)
    return template


def setup_submit(data_dump_file, inputs, args, cli_args):
    # Create analysis nodes
    analysis_nodes = []
    for idx in range(args.n_parallel):
        node = AnalysisNode(data_dump_file, inputs, idx, args, cli_args)
        node.write()
        analysis_nodes.append(node)

    if len(analysis_nodes) > 1:
        final_analysis_node = MergeNodes(analysis_nodes, inputs, args)
        final_analysis_node.write()
    else:
        final_analysis_node = analysis_nodes[0]

    bash_script = f"{inputs.submit_directory}/bash_{inputs.label}.sh"
    with open(bash_script, "w+") as ff:
        dependent_job_ids = []
        for ii, node in enumerate(analysis_nodes):
            print(f"jid{ii}=$(sbatch {node.filename})", file=ff)
            dependent_job_ids.append(f"${{jid{ii}##* }}")
        if len(analysis_nodes) > 1:
            print(
                f"sbatch --dependency=afterok:{':'.join(dependent_job_ids)} "
                f"{final_analysis_node.filename}",
                file=ff,
            )
        print('squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"', file=ff)

    return bash_script


class BaseNode(object):
    def __init__(self, inputs: Namespace, args: Namespace):
        self.inputs = inputs
        self.args = args

        self.nodes = self.args.nodes
        self.ntasks_per_node = self.args.ntasks_per_node
        self.time = self.args.time
        self.mem_per_cpu = self.args.mem_per_cpu

    def get_contents(self, command):
        template = load_template(TEMPLATE_SLURM)
        log_file = f"{self.logs}/{self.job_name}_%j.log"

        if self.args.slurm_extra_lines is not None:
            slurm_extra_lines = "\n".join(
                [f"#SBATCH --{lin}" for lin in self.args.slurm_extra_lines.split()]
            )
        else:
            slurm_extra_lines = ""

        if self.mem_per_cpu is not None:
            slurm_extra_lines += f"\n#SBATCH --mem-per-cpu={self.mem_per_cpu}"

        if self.args.extra_lines:
            bash_extra_lines = self.args.extra_lines.split(";")
            bash_extra_lines = "\n".join([line.strip() for line in bash_extra_lines])
        else:
            bash_extra_lines = ""

        file_contents = template.render(
            job_name=self.job_name,
            nodes=self.nodes,
            ntasks_per_node=self.ntasks_per_node,
            time=self.time,
            log_file=log_file,
            mem_per_cpu=self.mem_per_cpu,
            slurm_extra_lines=slurm_extra_lines,
            bash_extra_lines=bash_extra_lines,
            command=command,
        )

        return file_contents

    def write(self):
        content = self.get_contents()
        with open(self.filename, "w+") as f:
            print(content, file=f)


class AnalysisNode(BaseNode):
    def __init__(self, data_dump_file, inputs, idx, args, cli_args):
        super().__init__(inputs, args)
        self.data_dump_file = data_dump_file

        self.idx = idx
        self.filename = (
            f"{self.inputs.submit_directory}/"
            f"analysis_{self.inputs.label}_{self.idx}.sh"
        )
        self.job_name = f"{self.idx}_{self.inputs.label}"
        self.logs = self.inputs.data_analysis_log_directory

        analysis_parser = create_analysis_parser(sampler=self.args.sampler)
        self.analysis_args, _ = analysis_parser.parse_known_args(args=cli_args)
        self.analysis_args.data_dump = self.data_dump_file

    @property
    def executable(self):
        if self.args.sampler == "dynesty":
            return "parallel_bilby_analysis"
        else:
            raise ValueError(
                f"Unable to determine sampler to use from {self.args.sampler}"
            )

    @property
    def label(self):
        return f"{self.inputs.label}_{self.idx}"

    @property
    def output_filename(self):
        return (
            f"{self.inputs.result_directory}/"
            f"{self.inputs.label}_{self.idx}_result.{self.analysis_args.result_format}"
        )

    def get_contents(self):
        command = f"mpirun {self.executable} {self.get_run_string()}"
        return super().get_contents(command=command)

    def get_run_string(self):
        run_list = [f"{self.data_dump_file}"]
        for key, val in vars(self.analysis_args).items():
            if key in ["data_dump", "label", "outdir", "sampling_seed"]:
                continue
            input_val = getattr(self.args, key)
            if val != input_val:
                key = key.replace("_", "-")
                if input_val is True:
                    # For flags only add the flag
                    run_list.append(f"--{key}")
                elif isinstance(input_val, list):
                    # For lists add each entry individually
                    for entry in input_val:
                        run_list.append(f"--{key} {entry}")
                else:
                    run_list.append(f"--{key} {input_val}")

        run_list.append(f"--label {self.label}")
        run_list.append(f"--sampling-seed {self.inputs.sampling_seed + self.idx}")
        run_list.append(f"--outdir {abspath(self.inputs.result_directory)}")

        return " ".join(run_list)


class MergeNodes(BaseNode):
    def __init__(self, analysis_nodes, inputs, args):
        super().__init__(inputs, args)
        self.analysis_nodes = analysis_nodes
        self.job_name = f"merge_{self.inputs.label}"
        self.nodes = 1
        self.ntasks_per_node = 1
        self.time = "1:00:00"
        self.mem_per_cpu = "16GB"
        self.logs = self.inputs.data_analysis_log_directory
        self.filename = f"{self.inputs.submit_directory}/merge_{self.inputs.label}.sh"

    @property
    def file_list(self):
        return " ".join([node.output_filename for node in self.analysis_nodes])

    @property
    def merged_result_label(self):
        return f"{self.inputs.label}_merged"

    def get_contents(self):
        command = []
        command.append(f"bilby_result -r {self.file_list}")
        command.append("--merge")
        command.append(f"--label {self.merged_result_label}")
        command.append(f"--outdir {self.inputs.result_directory}")
        command.append(f"-e {self.args.result_format}")
        command = " ".join(command)
        return super().get_contents(command=command)
