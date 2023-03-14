import bilby
import bilby_pipe.data_generation

from .shared import _create_base_parser

logger = bilby.core.utils.logger


def remove_argument_from_parser(parser, arg):
    for action in parser._actions:
        if action.dest == arg.replace("-", "_"):
            try:
                parser._handle_conflict_resolve(None, [("--" + arg, action)])
            except ValueError as e:
                logger.warning(f"Error removing {arg}: {e}")
    logger.debug(f"Request to remove arg {arg} from bilby_pipe args, but arg not found")


def _add_slurm_settings_to_parser(parser):
    slurm_group = parser.add_argument_group(title="Slurm Settings")
    slurm_group.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes to use (default 1)"
    )
    slurm_group.add_argument(
        "--ntasks-per-node",
        type=int,
        default=2,
        help="Number of tasks per node (default 2)",
    )
    slurm_group.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        help="Maximum wall time (defaults to 24:00:00)",
    )
    slurm_group.add_argument(
        "--mem-per-cpu",
        type=str,
        default="2G",
        help="Memory per CPU (defaults to 2GB)",
    )
    slurm_group.add_argument(
        "--extra-lines",
        type=str,
        default=None,
        help="Additional lines, separated by ';', use for setting up conda env or module imports",
    )
    slurm_group.add_argument(
        "--slurm-extra-lines",
        type=str,
        default=None,
        help="additional slurm args (args that need #SBATCH in front) of the form arg=val separated by sapce",
    )
    return parser


def _create_reduced_bilby_pipe_parser():
    bilby_pipe_parser = bilby_pipe.parser.create_parser()
    bilby_pipe_arguments_to_ignore = [
        "version",
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
        "plot-calibration",
        "plot-corner",
        "plot-format",
        "plot-marginal",
        "plot-skymap",
        "plot-waveform",
    ]
    for arg in bilby_pipe_arguments_to_ignore:
        remove_argument_from_parser(bilby_pipe_parser, arg)

    bilby_pipe_parser.add_argument(
        "--sampler",
        choices=["dynesty"],
        default="dynesty",
        type=str,
        help="The parallelised sampler to use, defaults to dynesty",
    )
    return bilby_pipe_parser


def add_extra_args_from_bilby_pipe_namespace(cli_args, parallel_bilby_args):
    """
    :param args: args from parallel_bilby
    :return: Namespace argument object
    """
    pipe_args, _ = bilby_pipe.data_generation.parse_args(
        cli_args, bilby_pipe.data_generation.create_generation_parser()
    )
    for key, val in vars(pipe_args).items():
        if key not in parallel_bilby_args:
            setattr(parallel_bilby_args, key, val)
    return parallel_bilby_args


def create_generation_parser():
    """Parser for parallel_bilby_generation"""
    parser = _create_base_parser(sampler="all")
    bilby_pipe_parser = _create_reduced_bilby_pipe_parser()
    generation_parser = bilby_pipe.parser.BilbyArgParser(
        prog="parallel_bilby_generation",
        usage=__doc__,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        parents=[parser, bilby_pipe_parser],
        add_help=False,
    )
    generation_parser = _add_slurm_settings_to_parser(generation_parser)
    return generation_parser


def parse_generation_args(parser, cli_args=[""], as_namespace=False):
    """
    Returns dictionary of arguments, as specified in the
    parser.

    If no cli_args arguments are specified, returns the default arguments
    (by running the parser with no ini file and no CLI arguments)

    Parameters
    ----------
    parser: generation-parser
    cli_args: list of strings (default: [""])
        List of arguments to be parsed. If empty, returns default arguments
    as_namespace: bool (default False)
        If True, returns the arguments as a Namespace object. If False, returns a dict

    Returns
    -------
    args: dict or Namespace

    """

    args = parser.parse_args(args=cli_args)
    args = add_extra_args_from_bilby_pipe_namespace(cli_args, args)
    if as_namespace:
        return args
    return vars(args)
