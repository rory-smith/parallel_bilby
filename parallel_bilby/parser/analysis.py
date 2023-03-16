import argparse

from .shared import _create_base_parser


def create_analysis_parser(sampler="dynesty"):
    """Parser for parallel_bilby_analysis"""
    parser = _create_base_parser(sampler=sampler)

    analysis_parser = argparse.ArgumentParser(
        prog="parallel_bilby_analysis", parents=[parser]
    )
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
    analysis_parser.add_argument(
        "--result-format", default="hdf5", type=str, help="Format to save the result"
    )
    return analysis_parser


def parse_analysis_args(parser, cli_args=[""]):
    """Parse the command line arguments for parallel_bilby_analysis"""
    args = parser.parse_args(args=cli_args)

    if args.walks > args.maxmcmc:
        raise ValueError(
            f"You have maxmcmc ({args.maxmcmc}) > walks ({args.walks}, minimum mcmc)"
        )
    if args.nact < 1:
        raise ValueError(f"Your nact ({args.nact}) < 1 (must be >= 1)")

    return args
