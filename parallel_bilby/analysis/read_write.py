import os
import shutil

import bilby
import dill
import numpy as np
from pandas import DataFrame

from bilby.core.utils import logger
from ..utils import stopwatch, safe_file_dump


@stopwatch
def write_current_state(sampler, resume_file, sampling_time, rotate=False):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    rotate: bool
        If resume_file already exists, first make a backup file (ending in '.bk').
    """
    print("")
    logger.info("Start checkpoint writing")
    if rotate and os.path.isfile(resume_file):
        resume_file_bk = resume_file + ".bk"
        logger.info(f"Backing up existing checkpoint file to {resume_file_bk}")
        shutil.copyfile(resume_file, resume_file_bk)
    sampler.kwargs["sampling_time"] = sampling_time
    if dill.pickles(sampler):
        safe_file_dump(sampler, resume_file, dill)
        logger.info(f"Written checkpoint file {resume_file}")
    else:
        logger.warning("Cannot write pickle resume file!")


def write_sample_dump(sampler, samples_file, search_parameter_keys):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    """

    ln_weights = sampler.saved_logwt - sampler.saved_logz[-1]
    weights = np.exp(ln_weights)
    samples = bilby.core.result.rejection_sample(np.array(sampler.saved_v), weights)
    nsamples = len(samples)

    # If we don't have enough samples, don't dump them
    if nsamples < 100:
        return

    logger.info(f"Writing {nsamples} current samples to {samples_file}")
    df = DataFrame(samples, columns=search_parameter_keys)
    df.to_csv(samples_file, index=False, header=True, sep=" ")


@stopwatch
def read_saved_state(resume_file, continuing=True):
    """
    Read a saved state of the sampler to disk.

    The required information to reconstruct the state of the run is read from a
    pickle file.

    Parameters
    ----------
    resume_file: str
        The path to the resume file to read

    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was successfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """

    if os.path.isfile(resume_file):
        logger.info(f"Reading resume file {resume_file}")
        with open(resume_file, "rb") as file:
            sampler = dill.load(file)
            if sampler.added_live and continuing:
                sampler._remove_live_points()
            sampler.nqueue = -1
            sampler.rstate = np.random
            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        logger.info(f"Resume file {resume_file} does not exist.")
        return False, 0
