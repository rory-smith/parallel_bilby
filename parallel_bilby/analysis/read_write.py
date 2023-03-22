import logging
import os
import shutil
import time

import bilby
import dill
import dynesty
import numpy as np
from bilby.core.utils import logger
from pandas import DataFrame

from ..utils import safe_file_dump, stopwatch
from .likelihood import reorder_loglikelihoods


def _get_time_since_file_last_modified(file_path):
    """Returns the time since the file was last modified in human-readable format

    Parameters
    ----------
    file_path: str
        Path to the file

    Returns
    -------
    time_since_last_modified: str
    """
    if not os.path.exists(file_path):
        return ""
    seconds = time.time() - os.path.getmtime(file_path)
    d, remainder = divmod(seconds, 86400)
    hr, remainder = divmod(remainder, 3600)
    m, s = divmod(remainder, 60)
    strtime = f"{m:02.0f}m {s:02.0f}s"
    strtime = f"{hr:02.0f}h {strtime}" if hr > 0 else strtime
    strtime = f"{d:02.0f} days, {strtime}" if d > 0 else strtime
    return strtime


@stopwatch(log_level=logging.INFO)
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

    time_since_save = _get_time_since_file_last_modified(resume_file)
    logger.info(
        "Start checkpoint writing" + f" (last checkpoint {time_since_save} ago)"
        if time_since_save
        else ""
    )
    if rotate and os.path.isfile(resume_file):
        resume_file_bk = resume_file + ".bk"
        logger.info(f"Backing up existing checkpoint file to {resume_file_bk}")
        shutil.copyfile(resume_file, resume_file_bk)
    sampler.kwargs["sampling_time"] = sampling_time

    # Get random state and package it into the resume object
    sampler.kwargs["random_state"] = sampler.rstate.bit_generator.state

    if dill.pickles(sampler):
        safe_file_dump(sampler, resume_file, dill)
        logger.info(f"Written checkpoint file {resume_file}")
    else:
        logger.warning("Cannot write pickle resume file!")

    # Delete the random state so that the object is unchanged
    del sampler.kwargs["random_state"]


def write_sample_dump(sampler, samples_file, search_parameter_keys):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    """

    ln_weights = sampler.saved_run.D["logwt"] - sampler.saved_run.D["logz"][-1]
    weights = np.exp(ln_weights)
    samples = bilby.core.result.rejection_sample(
        np.array(sampler.saved_run.D["v"]), weights
    )
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

            # Create random number generator and restore state
            # from file, then remove it from kwargs because it
            # is not useful after the generator has been cycled
            sampler.rstate = np.random.Generator(np.random.PCG64())
            sampler.rstate.bit_generator.state = sampler.kwargs.pop("random_state")

            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        logger.info(f"Resume file {resume_file} does not exist.")
        return False, 0


def format_result(
    run,
    data_dump,
    out,
    weights,
    nested_samples,
    sampler_kwargs,
    sampling_time,
    rejection_sample_posterior=True,
):
    """
    Packs the variables from the run into a bilby result object

    Parameters
    ----------
    run: AnalysisRun
        Parallel Bilby run object
    data_dump: str
        Path to the *_data_dump.pickle file
    out: dynesty.results.Results
        Results from the dynesty sampler
    weights: numpy.ndarray
        Array of weights for the points
    nested_samples: pandas.core.frame.DataFrame
        DataFrame of the weights and likelihoods
    sampler_kwargs: dict
        Dictionary of keyword arguments for the sampler
    sampling_time: float
        Time in seconds spent sampling
    rejection_sample_posterior: bool
        Whether to generate the posterior samples by rejection sampling the
        nested samples or resampling with replacement

    Returns
    -------
    result: bilby.core.result.Result
        result object with values written into its attributes
    """

    result = bilby.core.result.Result(
        label=run.label, outdir=run.outdir, search_parameter_keys=run.sampling_keys
    )
    result.priors = run.priors
    result.nested_samples = nested_samples
    result.meta_data = run.data_dump["meta_data"]
    result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
    result.meta_data["data_dump"] = data_dump
    result.meta_data["likelihood"] = run.likelihood.meta_data
    result.meta_data["sampler_kwargs"] = run.init_sampler_kwargs
    result.meta_data["run_sampler_kwargs"] = sampler_kwargs
    result.meta_data["injection_parameters"] = run.injection_parameters
    result.injection_parameters = run.injection_parameters

    if rejection_sample_posterior:
        keep = weights > np.random.uniform(0, max(weights), len(weights))
        result.samples = out.samples[keep]
        result.log_likelihood_evaluations = out.logl[keep]
        logger.info(
            f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples"
        )
    else:
        result.samples = dynesty.utils.resample_equal(out.samples, weights)
        result.log_likelihood_evaluations = reorder_loglikelihoods(
            unsorted_loglikelihoods=out.logl,
            unsorted_samples=out.samples,
            sorted_samples=result.samples,
        )
        logger.info("Resampling nested samples to posterior samples in place.")

    result.log_evidence = out.logz[-1] + run.likelihood.noise_log_likelihood()
    result.log_evidence_err = out.logzerr[-1]
    result.log_noise_evidence = run.likelihood.noise_log_likelihood()
    result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
    result.sampling_time = sampling_time
    result.num_likelihood_evaluations = np.sum(out.ncall)

    result.samples_to_posterior(likelihood=run.likelihood, priors=result.priors)
    return result
