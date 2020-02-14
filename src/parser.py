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


base_parser = argparse.ArgumentParser("base", add_help=False)

dynesty_settings_parser = base_parser.add_argument_group(title="Dynesty Settings")
dynesty_settings_parser.add_argument(
    "-n", "--nlive", default=1000, type=int, help="Number of live points"
)
dynesty_settings_parser.add_argument(
    "--dlogz",
    default=0.1,
    type=float,
    help="Stopping criteria: remaining evidence, (default=0.1)",
)
dynesty_settings_parser.add_argument(
    "--n-effective",
    default=5000,
    type=int,
    help="Stopping criteria: effective number of samples, (default=5000)",
)
dynesty_settings_parser.add_argument(
    "--dynesty-sample",
    default="rwalk",
    type=str,
    help="Dynesty sampling method (default=rwalk). Note, the dynesty rwalk "
    "method is overwritten by parallel bilby for an optimised version ",
)
dynesty_settings_parser.add_argument(
    "--dynesty-bound",
    default="multi",
    type=str,
    help="Dynesty bounding method (default=multi)",
)
dynesty_settings_parser.add_argument(
    "--walks", default=100, type=int, help="Minimum number of walks, defaults to 100"
)
dynesty_settings_parser.add_argument(
    "--maxmcmc",
    default=5000,
    type=int,
    help="Maximum number of walks, defaults to 5000",
)
dynesty_settings_parser.add_argument(
    "--nact",
    default=5,
    type=int,
    help="Number of autocorrelation times to take, defaults to 5",
)
dynesty_settings_parser.add_argument(
    "--min-eff",
    default=10,
    type=float,
    help="The minimum efficiency at which to switch from uniform sampling.",
)
dynesty_settings_parser.add_argument(
    "--facc", default=0.5, type=float, help="See dynesty.NestedSampler"
)
dynesty_settings_parser.add_argument(
    "--vol-dec", default=0.5, type=float, help="See dynesty.NestedSampler"
)
dynesty_settings_parser.add_argument(
    "--vol-check", default=8, type=float, help="See dynesty.NestedSampler"
)
dynesty_settings_parser.add_argument(
    "--enlarge", default=1.5, type=float, help="See dynesty.NestedSampler"
)
dynesty_settings_parser.add_argument(
    "--n-check-point", default=100000, type=int, help="Steps to take before checkpoint"
)

misc_settings_parser = base_parser.add_argument_group(title="Misc. Settings")
misc_settings_parser.add_argument(
    "--bilby-zero-likelihood-mode", default=False, action="store_true"
)
misc_settings_parser.add_argument(
    "--rand-seed",
    type=int,
    default=1234,
    help="Random seed: important for reproducible resampling",
)
misc_settings_parser.add_argument(
    "-c", "--clean", action="store_true", help="Run clean: ignore any resume files"
)
misc_settings_parser.add_argument(
    "--no-plot", action="store_true", help="If true, don't generate check-point plots"
)

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
]
for arg in bilby_pipe_arguments_to_ignore:
    remove_argument_from_parser(bilby_pipe_parser, arg)

generation_parser = bilby_pipe.parser.BilbyArgParser(
    usage=__doc__,
    ignore_unknown_config_file_keys=False,
    allow_abbrev=False,
    parents=[base_parser, bilby_pipe_parser],
    add_help=False,
)
slurm_settings = generation_parser.add_argument_group(title="Slurm Settings")
slurm_settings.add_argument(
    "--nodes", type=int, required=True, help="Number of nodes to use"
)
slurm_settings.add_argument(
    "--ntasks-per-node", type=int, required=True, help="Number of tasks per node"
)
slurm_settings.add_argument(
    "--time",
    type=str,
    default="24:00:00",
    required=True,
    help="Maximum wall time (defaults to 24:00:00)",
)
slurm_settings.add_argument(
    "--mem-per-cpu", type=str, default="4000", help="Memory per CPU (defaults to 4000)"
)
slurm_settings.add_argument(
    "--extra-lines",
    type=str,
    default=None,
    help="Additional lines, separated by ';', use for setting up conda env ",
)


analysis_parser = argparse.ArgumentParser("base", parents=[base_parser])
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
