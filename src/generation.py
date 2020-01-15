#!/usr/bin/env python
"""
Generate/prepare data, likelihood, and priors for parallel runs
"""
import pickle

import bilby
from bilby_pipe.data_generation import (DataGenerationInput,
                                        create_generation_parser, parse_args)

from .utils import get_cli_args
from .parser import generation_parser
from . import slurm

logger = bilby.core.utils.logger


def add_extra_args_from_bilby_pipe_namespace(args):
    pipe_args, _ = parse_args(
        get_cli_args(), create_generation_parser())
    for key, val in vars(pipe_args).items():
        if key not in args:
            setattr(args, key, val)
    return args


def main():
    args = generation_parser.parse_args()
    args = add_extra_args_from_bilby_pipe_namespace(args)
    inputs = DataGenerationInput(args, [])

    ifo_list = inputs.interferometers
    data_dir = inputs.data_directory
    label = inputs.label
    ifo_list.plot_data(outdir=data_dir, label=label)

    logger.info(
        "Setting up likelihood with marginalizations: "
        f"distance={args.distance_marginalization} "
        f"time={args.time_marginalization} "
        f"phase={args.phase_marginalization} ")

    # This is done before instantiating the likelihood so that it is the full prior
    prior_file = f"{data_dir}/{label}_prior.json"
    inputs.priors.to_json(outdir=data_dir, label=label)

    # We build the likelihood here to ensure the distance marginalization exist
    # before sampling
    inputs.likelihood

    data_dump_file = f"{data_dir}/{label}_data_dump.pickle"
    data_dump = dict(
        waveform_generator=inputs.waveform_generator, ifo_list=ifo_list,
        prior_file=prior_file, args=args, data_dump_file=data_dump_file)

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)

    slurm.setup_submit(data_dump_file, inputs, args)
    logger.info("Generation done: submit files stored in {}"
                .format(inputs.submit_directory))
