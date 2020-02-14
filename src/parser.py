import argparse

from numpy import inf

import bilby
import bilby_pipe

logger = bilby.core.utils.logger


def remove_argument_from_parser(parser, arg):
    for action in parser._actions:
        if action.dest == arg.replace("-", "_"):
            parser._handle_conflict_resolve(None, [("--" + arg, action)])
            return
    logger.warning(
        "Request to remove arg {} from bilby_pipe args, but arg not found".format(arg)
    )


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


def _create_base_parser():
    base_parser = argparse.ArgumentParser("base", add_help=False)
    base_parser = _add_dyesty_settings_to_parser(base_parser)
    base_parser = _add_misc_settings_to_parser(base_parser)
    return base_parser


def _add_dyesty_settings_to_parser(parser):
    dynesty_group = parser.add_argument_group(title="Dynesty Settings")
    dynesty_group.add_argument(
        "-n", "--nlive", default=1000, type=int, help="Number of live points"
    )
    dynesty_group.add_argument(
        "--dlogz",
        default=0.1,
        type=float,
        help="Stopping criteria: remaining evidence, (default=0.1)",
    )
    dynesty_group.add_argument(
        "--n-effective",
        default=inf,
        type=float,
        help="Stopping criteria: effective number of samples, (default=inf)",
    )
    dynesty_group.add_argument(
        "--dynesty-sample",
        default="rwalk",
        type=str,
        help="Dynesty sampling method (default=rwalk). Note, the dynesty rwalk "
        "method is overwritten by parallel bilby for an optimised version ",
    )
    dynesty_group.add_argument(
        "--dynesty-bound",
        default="multi",
        type=str,
        help="Dynesty bounding method (default=multi)",
    )
    dynesty_group.add_argument(
        "--walks",
        default=100,
        type=int,
        help="Minimum number of walks, defaults to 100",
    )
    dynesty_group.add_argument(
        "--maxmcmc",
        default=5000,
        type=int,
        help="Maximum number of walks, defaults to 5000",
    )
    dynesty_group.add_argument(
        "--nact",
        default=5,
        type=int,
        help="Number of autocorrelation times to take, defaults to 5",
    )
    dynesty_group.add_argument(
        "--min-eff",
        default=10,
        type=float,
        help="The minimum efficiency at which to switch from uniform sampling.",
    )
    dynesty_group.add_argument(
        "--facc", default=0.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--vol-dec", default=0.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--vol-check", default=8, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--enlarge", default=1.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--n-check-point",
        default=100000,
        type=int,
        help="Steps to take before checkpoint",
    )
    return parser


def _add_misc_settings_to_parser(parser):
    misc_group = parser.add_argument_group(title="Misc. Settings")
    misc_group.add_argument(
        "--bilby-zero-likelihood-mode", default=False, action="store_true"
    )
    misc_group.add_argument(
        "--sampling-seed",
        type=int,
        default=1234,
        help="Random seed for sampling, parallel runs will be incremented",
    )
    misc_group.add_argument(
        "-c", "--clean", action="store_true", help="Run clean: ignore any resume files"
    )
    misc_group.add_argument(
        "--no-plot",
        action="store_true",
        help="If true, don't generate check-point plots",
    )
    return parser


def _add_slurm_settings_to_parser(parser):
    slurm_group = parser.add_argument_group(title="Slurm Settings")
    slurm_group.add_argument(
        "--nodes", type=int, required=True, help="Number of nodes to use"
    )
    slurm_group.add_argument(
        "--ntasks-per-node", type=int, required=True, help="Number of tasks per node"
    )
    slurm_group.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        required=True,
        help="Maximum wall time (defaults to 24:00:00)",
    )
    slurm_group.add_argument(
        "--mem-per-cpu",
        type=str,
        default="1000",
        help="Memory per CPU (defaults to 1000 MB)",
    )
    slurm_group.add_argument(
        "--extra-lines",
        type=str,
        default=None,
        help="Additional lines, separated by ';', use for setting up conda env ",
    )
    return parser


def _create_reduced_bilby_pipe_parser():
    bilby_pipe_parser = bilby_pipe.parser.create_parser()
    bilby_pipe_arguments_to_ignore = [
        "accounting",
        "local",
        "local-generation",
        "local-plot",
        "request-memory",
        "request-memory-generation",
        "request-cpus",
        "singularity-image",
        "scheduler",
        "scheduler-args",
        "scheduler-module",
        "scheduler-env",
        "transfer-files",
        "online-pe",
        "osg",
        "email",
        "postprocessing-executable",
        "postprocessing-arguments",
        "sampler",
        "sampling-seed",
        "sampler-kwargs",
        "n-parallel",
    ]
    for arg in bilby_pipe_arguments_to_ignore:
        remove_argument_from_parser(bilby_pipe_parser, arg)
    return bilby_pipe_parser


def create_generation_parser():
    """Parser for parallel_bilby_generation"""
    parser = _create_base_parser()
    bilby_pipe_parser = _create_reduced_bilby_pipe_parser()
    generation_parser = bilby_pipe.parser.BilbyArgParser(
        usage=__doc__,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        parents=[parser, bilby_pipe_parser],
        add_help=False,
    )
    generation_parser = _add_slurm_settings_to_parser(generation_parser)
    return generation_parser


def create_analysis_parser():
    """Parser for parallel_bilby_analysis"""
    parser = _create_base_parser()
    analysis_parser = argparse.ArgumentParser("base", parents=[parser])
    analysis_parser.add_argument(
        "data_dump",
        type=str,
        help="The pickled data dump generated by parallel_bilby_analysis",
    )
    analysis_parser.add_argument(
        "--outdir", default=None, type=str, help="Outdir to overwrite input label"
    )
    analysis_parser.add_argument(
        "--label", default=None, type=str, help="Label to overwrite input label"
    )
    return analysis_parser
