#!/usr/bin/env python
"""
Module to generate/prepare data, likelihood, and priors for parallel runs.

This will create a directory structure for your parallel runs to store the
output files, logs and plots. It will also generate a `data_dump` that stores
information on the run settings and data to be analysed.
"""
import pickle
import shutil
import subprocess

import bilby
import bilby_pipe
import dynesty
from bilby_pipe import data_generation as bilby_pipe_datagen
from bilby_pipe.data_generation import parse_args

from . import __version__, slurm
from .parser import create_generation_parser
from .utils import get_cli_args

logger = bilby.core.utils.logger

generation_parser = create_generation_parser()


def add_extra_args_from_bilby_pipe_namespace(args):
    """
    :param args: args from parallel_bilby
    :return: Namespace argument object
    """
    pipe_args, _ = parse_args(
        get_cli_args(), bilby_pipe_datagen.create_generation_parser()
    )
    for key, val in vars(pipe_args).items():
        if key not in args:
            setattr(args, key, val)
    return args


def main():
    cli_args = get_cli_args()
    args = generation_parser.parse_args(args=cli_args)
    args = add_extra_args_from_bilby_pipe_namespace(args)
    inputs = bilby_pipe_datagen.DataGenerationInput(args, [])
    inputs.log_directory = None
    shutil.rmtree(inputs.data_generation_log_directory)  # Hack to remove unused dir

    ifo_list = inputs.interferometers
    data_dir = inputs.data_directory
    label = inputs.label
    ifo_list.plot_data(outdir=data_dir, label=label)

    logger.info(
        "Setting up likelihood with marginalizations: "
        f"distance={args.distance_marginalization} "
        f"time={args.time_marginalization} "
        f"phase={args.phase_marginalization} "
    )

    # This is done before instantiating the likelihood so that it is the full prior
    prior_file = f"{data_dir}/{label}_prior.json"
    inputs.priors.to_json(outdir=data_dir, label=label)

    # We build the likelihood here to ensure the distance marginalization exist
    # before sampling
    inputs.likelihood

    data_dump_file = f"{data_dir}/{label}_data_dump.pickle"

    meta_data = dict(
        config_file=args.ini,
        data_dump_file=data_dump_file,
        bilby_version=bilby.__version__,
        bilby_pipe_version=bilby_pipe.__version__,
        parallel_bilby_version=__version__,
        dynesty_version=dynesty.__version__,
    )
    logger.info("Initial meta_data = {}".format(meta_data))

    data_dump = dict(
        waveform_generator=inputs.waveform_generator,
        ifo_list=ifo_list,
        prior_file=prior_file,
        args=args,
        data_dump_file=data_dump_file,
        meta_data=meta_data,
        injection_parameters=inputs.injection_parameters,
    )

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)

    bash_file = slurm.setup_submit(data_dump_file, inputs, args)
    if args.submit:
        subprocess.run(["bash {}".format(bash_file)], shell=True)
    else:
        logger.info("Setup complete, now run:\n $ bash {}".format(bash_file))
