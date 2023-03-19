import argparse

import bilby
import bilby_pipe
from numpy import inf

from ..utils import get_version_information

logger = bilby.core.utils.logger

__version__ = get_version_information()


class StoreBoolean(argparse.Action):
    """argparse class for robust handling of booleans with configargparse

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


def _create_base_parser(sampler="dynesty"):
    base_parser = argparse.ArgumentParser("base", add_help=False)
    base_parser.add(
        "--version",
        action="version",
        version=f"%(prog)s={__version__}\nbilby={bilby.__version__}",
    )
    if sampler in ["all", "dynesty"]:
        base_parser = _add_dynesty_settings_to_parser(base_parser)
    base_parser = _add_misc_settings_to_parser(base_parser)
    return base_parser


def _add_dynesty_settings_to_parser(parser):
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
        default="acceptance-walk",
        type=str,
        help="Dynesty sampling method (default=acceptance-walk). "
        "Note, the dynesty rwalk method is overwritten by parallel bilby for an optimised version ",
    )
    dynesty_group.add_argument(
        "--dynesty-bound",
        default="live",
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
        "--proposals",
        type=bilby_pipe.utils.nonestr,
        action="append",
        default=None,
        help="The jump proposals to use, the options are 'diff' and 'volumetric'",
    )
    dynesty_group.add_argument(
        "--maxmcmc",
        default=5000,
        type=int,
        help="Maximum number of walks, defaults to 5000",
    )
    dynesty_group.add_argument(
        "--nact",
        default=2,
        type=int,
        help="Number of autocorrelation times to take, defaults to 2",
    )
    dynesty_group.add_argument(
        "--naccept",
        default=60,
        type=int,
        help="The average number of accepted steps per MCMC chain, defaults to 60",
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
        "--enlarge", default=1.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--n-check-point",
        default=1000,
        type=int,
        help="Steps to take before attempting checkpoint",
    )
    dynesty_group.add_argument(
        "--max-its",
        default=10**10,
        type=int,
        help="Maximum number of iterations to sample for (default=1.e10)",
    )
    dynesty_group.add_argument(
        "--max-run-time",
        default=1.0e10,
        type=float,
        help="Maximum time to run for (default=1.e10 s)",
    )
    dynesty_group.add_argument(
        "--fast-mpi",
        default=False,
        type=bool,
        help="Fast MPI communication pattern (default=False)",
    )
    dynesty_group.add_argument(
        "--mpi-timing",
        default=False,
        type=bool,
        help="Print MPI timing when finished (default=False)",
    )
    dynesty_group.add_argument(
        "--mpi-timing-interval",
        default=0,
        type=int,
        help="Interval to write timing snapshot to disk (default=0 -- disabled)",
    )
    dynesty_group.add_argument(
        "--nestcheck",
        default=False,
        action="store_true",
        help=(
            "Save a 'nestcheck' pickle in the outdir (default=False). "
            "This pickle stores a `nestcheck.data_processing.process_dynesty_run` "
            "object, which can be used during post processing to compute the "
            "implementation and bootstrap errors explained by Higson et al (2018) "
            "in “Sampling Errors In Nested Sampling Parameter Estimation”."
        ),
    )
    dynesty_group.add_argument(
        "--rejection-sample-posterior",
        default=True,
        action=StoreBoolean,
        help=(
            "Whether to generate the posterior samples by rejection sampling the "
            "nested samples or resampling with replacement"
        ),
    )
    return parser


def _add_misc_settings_to_parser(parser):
    misc_group = parser.add_argument_group(title="Misc. Settings")
    misc_group.add_argument(
        "--bilby-zero-likelihood-mode", default=False, action="store_true"
    )
    misc_group.add_argument(
        "--sampling-seed",
        type=bilby_pipe.utils.noneint,
        default=None,
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
    misc_group.add_argument(
        "--do-not-save-bounds-in-resume",
        default=True,
        action="store_true",
        help=(
            "If true, do not store bounds in the resume file. This can make "
            "resume files large (~GB)"
        ),
    )
    misc_group.add_argument(
        "--check-point-deltaT",
        default=3600,
        type=float,
        help="Write a checkpoint resume file and diagnostic plots every deltaT [s]. Default: 1 hour.",
    )
    misc_group.add_argument(
        "--rotate-checkpoints",
        action="store_true",
        help="If true, backup checkpoint before overwriting (ending in '.bk').",
    )
    return parser
