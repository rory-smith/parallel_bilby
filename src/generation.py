#!/usr/bin/env python
"""
Generate/prepare data, likelihood, and priors for parallel runs
"""
import argparse
import pickle

import bilby
from bilby_pipe.data_generation import (DataGenerationInput,
                                        create_generation_parser, parse_args)
from bilby_pipe.parser import BilbyArgParser
from bilby_pipe.utils import convert_string_to_dict

from .utils import get_cli_args

logger = bilby.core.utils.logger


class StoreBoolean(argparse.Action):
    """ argparse class for robust handling of booleans with configargparse

    When using configargparse, if the argument is setup with
    action="store_true", but the default is set to True, then there is no way,
    in the config file to switch the parameter off. To resolve this, this class
    handles the boolean properly.

    """

    def __call__(self, parser, namespace, value, option_string=None):
        value = str(value).lower()
        if value in ["true"]:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)


def get_args():
    parser = BilbyArgParser(
        usage=__doc__, ignore_unknown_config_file_keys=False,
        allow_abbrev=False
    )
    parser.add(
        "ini", type=str, is_config_file=True, help="Configuration ini file")
    parser.add_argument(
        "-l", "--label", help="Label for the data", required=True, type=str)
    parser.add_argument(
        "-o", "--outdir", default="processed_data",
        help="Outdir for the processed data", type=str)
    parser.add_argument(
        "-t", "--trigger-time", required=True, help="Trigger time", type=float)
    parser.add_argument(
        "-d", "--duration", required=True, help="Segment duration", type=float)
    parser.add(
        "--deltaT",
        type=float,
        default=0.2,
        help=(
            "The symmetric width (in s) around the trigger time to"
            " search over the coalesence time"
        ),
    )
    parser.add_argument(
        "--detectors",
        action="append",
        help=(
            "The names of detectors to use. If given in the ini file, "
            "detectors are specified by `detectors=[H1, L1]`. If given "
            "at the command line, as `--detectors H1 --detectors L1`"
        ),
    )
    parser.add_argument(
        "--data-dict", type=convert_string_to_dict, required=False, default=None,
        help="Dictionary of paths to the data to analyse, e.g. {H1:data.gwf}")
    parser.add_argument(
        "--channel-dict", type=convert_string_to_dict, required=False,
        help=("Dictionary of channel names for each data file, used when data-"
              "dict points to gwf files"))
    parser.add_argument(
        "--psd-dict", type=convert_string_to_dict, required=True,
        help="Dictionary of paths to the relevant PSD files for each data file"
    )
    parser.add_argument(
        "--prior-file", type=str, required=True,
        help="Path to the Bilby prior file")
    parser.add_argument(
        "--convert-to-flat-in-component-mass", action="store_true",
        help=("If true (default, False) resample to the LALInference "
              "flat-in-component mass prior"))
    parser.add_argument(
        "--waveform-approximant", type=str, required=True,
        help="Name of the waveform approximant")
    parser.add_argument(
        "--reference-frequency", default=20, help="The reference frequency",
        type=float)
    parser.add_argument(
        "--sampling-frequency", default=4096, help="The sampling frequency",
        type=float)
    parser.add_argument(
        "--minimum-frequency", default=20, help="The minimum frequency",
        type=float)
    parser.add_argument(
        "--maximum-frequency", default=2048, help="The maxmimum frequency",
        type=float)
    parser.add(
        "--calibration-model",
        default=None,
        choices=["CubicSpline"],
        help="Choice of calibration model, if None, no calibration is used",
        type=str,
    )
    parser.add(
        "--spline-calibration-envelope-dict",
        type=convert_string_to_dict,
        default=None,
        help=("Dictionary of paths to the spline calibration envelope files"),
    )
    parser.add(
        "--spline-calibration-nodes",
        type=int,
        default=10,
        help=("Number of calibration nodes"),
    )
    parser.add(
        "--distance-marginalization",
        action=StoreBoolean,
        required=True,
        help="Bool. If true, use a distance-marginalized likelihood",
    )
    parser.add(
        "--distance-marginalization-lookup-table",
        default=None,
        type=str,
        help="Path to the distance-marginalization lookup table",
    )
    parser.add(
        "--phase-marginalization",
        action=StoreBoolean,
        required=True,
        help="Bool. If true, use a phase-marginalized likelihood",
    )
    parser.add(
        "--time-marginalization",
        action=StoreBoolean,
        required=True,
        help="Bool. If true, use a time-marginalized likelihood",
    )
    parser.add(
        "--jitter-time",
        action=StoreBoolean,
        default=True,
        help="Bool. If true, use a the jitter-time option",
    )
    parser.add(
        "--binary-neutron-star",
        action=StoreBoolean,
        default=False,
        help="If true, use a BNS source model function (i.e. with tides)",
    )
    args = parser.parse_args()

    return args


def add_extra_args_from_bilby_pipe_namespace(args):
    pipe_args, _ = parse_args(
        get_cli_args(), create_generation_parser())
    for key, val in vars(pipe_args).items():
        if key not in args:
            setattr(args, key, val)
    return args


def main():
    args = get_args()
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

    logger.info("Generation done: now run:\nmpirun parallel_bilby_analysis {}"
                .format(data_dump_file))
