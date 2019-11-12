"""Module to retrieve data given a trigger time."""

import os

import bilby
from bilby_pipe.data_generation import (DataGenerationInput,
                                        create_generation_parser, parse_args)

from .utils import get_cli_args

logger = bilby.core.utils.logger


def retrieve_data(cli_args_list):
    """Setup a bilby_pipe folder structure, and download and plot data.

    :return: dict
        A dict in the format of {IFO_KEY: path_to_ifo_strain_hdf5}
    """
    # get args and manually set some args
    args, _ = parse_args(cli_args_list, create_generation_parser())
    args.create_plots = True

    # get data
    data = DataGenerationInput(args, [])

    # save data
    data.save_data_dump()
    data_dict = __save_ifo_strain(data.interferometers, args.outdir)

    logger.info("Completed data retrieval.")
    return data_dict


def __save_ifo_strain(interferometers, outdir):
    """Save ifo strain and return dict with paths to IFO's strain file."""
    data_dict = {}
    for ifo in interferometers:
        # get strain data
        strain = ifo.strain_data.to_gwpy_timeseries()
        strain.name = 'Strain'
        save_path = os.path.join(outdir, f"data/{ifo.name}_strain.hdf5")
        strain.write(save_path)
        data_dict.update({ifo.name: save_path})
    logger.info(f"Strain data stored in {data_dict}")
    return data_dict


def main():
    """CLI tool for retrieving data."""
    cli_args_list = get_cli_args()
    if cli_args_list:
        retrieve_data(cli_args_list)
    else:
        raise ValueError("No args passed")
