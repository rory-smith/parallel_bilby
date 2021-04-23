#!/usr/bin/env python
"""
Module to generate/prepare data, likelihood, and priors for parallel runs.

This will create a directory structure for your parallel runs to store the
output files, logs and plots. It will also generate a `data_dump` that stores
information on the run settings and data to be analysed.
"""
import os
import pickle
import subprocess

import bilby
import bilby_pipe
import dynesty
import lalsimulation
from bilby_pipe import data_generation as bilby_pipe_datagen
from bilby_pipe.data_generation import parse_args

from . import __version__, slurm
from .parser import create_generation_parser
from .utils import get_cli_args


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


def write_complete_config_file(parser, args, inputs):
    """Wrapper function that uses bilby_pipe's complete config writer.

    Note: currently this function does not verify that the written complete config is
    identical to the source config

    :param parser: The argparse.ArgumentParser to parse user input
    :param args: The parsed user input in a Namespace object
    :param inputs: The bilby_pipe.input.Input object storing user args
    :return: None
    """
    inputs.request_cpus = 1
    inputs.sampler_kwargs = "{}"
    inputs.mpi_timing_interval = 0
    inputs.log_directory = None
    try:
        bilby_pipe.main.write_complete_config_file(parser, args, inputs)
    except AttributeError:
        # bilby_pipe expects the ini to have "online_pe" and some other non pBilby args
        pass


def create_generation_logger(outdir, label):
    logger = bilby.core.utils.logger
    bilby.core.utils.setup_logger(
        outdir=os.path.join(outdir, "log_data_generation"), label=label
    )
    bilby_pipe.data_generation.logger = logger
    return logger


def main():
    cli_args = get_cli_args()
    generation_parser = create_generation_parser()
    args = generation_parser.parse_args(args=cli_args)
    args = add_extra_args_from_bilby_pipe_namespace(args)
    logger = create_generation_logger(outdir=args.outdir, label=args.label)
    version_info = dict(
        bilby_version=bilby.__version__,
        bilby_pipe_version=bilby_pipe.__version__,
        parallel_bilby_version=__version__,
        dynesty_version=dynesty.__version__,
        lalsimulation_version=lalsimulation.__version__,
    )
    for package, version in version_info.items():
        logger.info(f"{package} version: {version}")

    inputs = bilby_pipe_datagen.DataGenerationInput(args, [])
    if inputs.likelihood_type == "ROQGravitationalWaveTransient":
        inputs.save_roq_weights()
    inputs.interferometers.plot_data(outdir=inputs.data_directory, label=inputs.label)
    logger.info(
        "Setting up likelihood with marginalizations: "
        f"distance={args.distance_marginalization} "
        f"time={args.time_marginalization} "
        f"phase={args.phase_marginalization} "
    )

    # This is done before instantiating the likelihood so that it is the full prior
    prior_file = f"{inputs.data_directory}/{inputs.label}_prior.json"
    inputs.priors.to_json(outdir=inputs.data_directory, label=inputs.label)

    # We build the likelihood here to ensure the distance marginalization exist
    # before sampling
    inputs.likelihood

    data_dump_file = f"{inputs.data_directory}/{inputs.label}_data_dump.pickle"

    meta_data = inputs.meta_data
    meta_data.update(
        dict(config_file=args.ini, data_dump_file=data_dump_file, **version_info)
    )
    logger.info("Initial meta_data = {}".format(meta_data))

    data_dump = dict(
        waveform_generator=inputs.waveform_generator,
        ifo_list=inputs.interferometers,
        prior_file=prior_file,
        args=args,
        data_dump_file=data_dump_file,
        meta_data=meta_data,
        injection_parameters=inputs.injection_parameters,
    )

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)

    write_complete_config_file(parser=generation_parser, args=args, inputs=inputs)
    logger.info(f"Complete ini written: {inputs.complete_ini_file}")

    bash_file = slurm.setup_submit(data_dump_file, inputs, args)
    if args.submit:
        subprocess.run(["bash {}".format(bash_file)], shell=True)
    else:
        logger.info("Setup complete, now run:\n $ bash {}".format(bash_file))
