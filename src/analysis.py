#!/usr/bin/env python
"""
Script to run paralell bilby using MPI
"""
import os
import sys
import argparse
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import bilby
import dynesty
from dynesty.plotting import traceplot
from dynesty import NestedSampler
from dynesty.utils import resample_equal
from schwimmbad import MPIPool
from pandas import DataFrame

import mpi4py
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False


logger = bilby.core.utils.logger


def main():
    """ Do nothing function to play nicely with MPI """
    pass


def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples,
                           sorted_samples):
    """ Reorders the stored log-likelihood after they have been reweighted

    This creates a sorting index by matching the reweights `result.samples`
    against the raw samples, then uses this index to sort the
    loglikelihoods

    Parameters
    ----------
    sorted_samples, unsorted_samples: array-like
        Sorted and unsorted values of the samples. These should be of the
        same shape and contain the same sample values, but in different
        orders
    unsorted_loglikelihoods: array-like
        The loglikelihoods corresponding to the unsorted_samples

    Returns
    -------
    sorted_loglikelihoods: array-like
        The loglikelihoods reordered to match that of the sorted_samples


    """

    idxs = []
    for ii in range(len(unsorted_loglikelihoods)):
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples,
                              axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match.")
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]


def write_checkpoint(
        sampler, resume_file, sampling_time, search_parameter_keys):
    """ Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    search_parameter_keys: list
        A list of the search parameter keys used in sampling (used for
        constructing checkpoint plots and pre-results)
    """
    print("")
    logger.info("Writing checkpoint file {}".format(resume_file))

    current_state = dict(
        unit_cube_samples=sampler.saved_u,
        physical_samples=sampler.saved_v,
        sample_likelihoods=sampler.saved_logl,
        sample_log_volume=sampler.saved_logvol,
        sample_log_weights=sampler.saved_logwt,
        cumulative_log_evidence=sampler.saved_logz,
        cumulative_log_evidence_error=sampler.saved_logzvar,
        cumulative_information=sampler.saved_h,
        id=sampler.saved_id,
        it=sampler.saved_it,
        nc=sampler.saved_nc,
        boundidx=sampler.saved_boundidx,
        bounditer=sampler.saved_bounditer,
        scale=sampler.saved_scale,
        sampling_time=sampling_time,
    )

    current_state.update(
        ncall=sampler.ncall, live_logl=sampler.live_logl,
        iteration=sampler.it - 1, live_u=sampler.live_u,
        live_v=sampler.live_v, nlive=sampler.nlive,
        live_bound=sampler.live_bound, live_it=sampler.live_it,
        added_live=sampler.added_live
    )

    # Try to save a set of current posterior samples
    try:
        weights = np.exp(current_state['sample_log_weights'] -
                         current_state['cumulative_log_evidence'][-1])

        current_state['posterior'] = resample_equal(
            np.array(current_state['physical_samples']), weights)
        current_state['search_parameter_keys'] = search_parameter_keys
    except ValueError:
        logger.debug("Unable to create posterior")

    with open(resume_file, 'wb') as file:
        pickle.dump(current_state, file)

    # Try to create a checkpoint traceplot
    try:
        fig = traceplot(sampler.results, labels=sampling_keys)[0]
        fig.tight_layout()
        fig.savefig(filename_trace)
        plt.close('all')
    except Exception:
        pass


def read_saved_state(resume_file, sampler):
    """
    Read a saved state of the sampler to disk.

    The required information to reconstruct the state of the run is read from a
    pickle file.

    Parameters
    ----------
    resume_file: str
        The path to the resume file to read
    sampler: `dynesty.NestedSampler`
        NestedSampler instance to reconstruct from the saved state.

    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was succesfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """

    if os.path.isfile(resume_file):
        logger.info("Reading resume file {}".format(resume_file))
        try:
            with open(resume_file, 'rb') as file:
                saved = pickle.load(file)
            logger.info(
                "Succesfuly read resume file {}".format(resume_file))
        except EOFError as e:
            logger.warning(
                "Resume file reading failed with error {}".format(e))
            return False, 0

        sampler.saved_u = list(saved['unit_cube_samples'])
        sampler.saved_v = list(saved['physical_samples'])
        sampler.saved_logl = list(saved['sample_likelihoods'])
        sampler.saved_logvol = list(saved['sample_log_volume'])
        sampler.saved_logwt = list(saved['sample_log_weights'])
        sampler.saved_logz = list(saved['cumulative_log_evidence'])
        sampler.saved_logzvar = list(saved['cumulative_log_evidence_error'])
        sampler.saved_id = list(saved['id'])
        sampler.saved_it = list(saved['it'])
        sampler.saved_nc = list(saved['nc'])
        sampler.saved_boundidx = list(saved['boundidx'])
        sampler.saved_bounditer = list(saved['bounditer'])
        sampler.saved_scale = list(saved['scale'])
        sampler.saved_h = list(saved['cumulative_information'])
        sampler.ncall = saved['ncall']
        sampler.live_logl = list(saved['live_logl'])
        sampler.it = saved['iteration'] + 1
        sampler.live_u = saved['live_u']
        sampler.live_v = saved['live_v']
        sampler.nlive = saved['nlive']
        sampler.live_bound = saved['live_bound']
        sampler.live_it = saved['live_it']
        sampler.added_live = saved['added_live']
        sampling_time = datetime.timedelta(
            seconds=saved['sampling_time']).total_seconds()
        return sampler, sampling_time

    else:
        logger.debug(
            "No resume file {}".format(resume_file))
        return False, 0


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "0"
os.environ["MPI_PER_NODE"] = "16"

parser = argparse.ArgumentParser("Analysis")
parser.add_argument(
    "data_dump", type=str,
    help="The pickled data dump generated by parallel_bilby_analysis")
parser.add_argument(
    "-l", "--label", default=None, nargs="*", help="Extra labels")
parser.add_argument(
    "-n", "--nlive", default=1000, type=int, help="Number of live points")
parser.add_argument(
    "--dlogz", default=0.1, type=float,
    help="Stopping criteria: remaining evidence, (default=0.1)")
parser.add_argument(
    "--n-effective", default=5000, type=float,
    help="Stopping criteria: effective number of samples, (default=5000)")
parser.add_argument(
    "--dynesty-sample", default="rwalk", type=str,
    help="Dynesty sampling method (default=rwalk)")
parser.add_argument(
    "--dynesty-walks", default=100, type=int,
    help="Number of walks")
parser.add_argument(
    "--n-check-point", default=10000, type=int,
    help="Number of walks")

input_args = parser.parse_args()

with open(input_args.data_dump, "rb") as file:
    data_dump = pickle.load(file)

priors = data_dump["priors"]
likelihood = data_dump["likelihood"]
args = data_dump["args"]
outdir = args.outdir
label = args.label
if input_args.label is not None:
    label += "_" + "_".join(input_args.label)
nlive = input_args.nlive


def prior_transform_function(u_array):
    return priors.rescale(sampling_keys, u_array)


def likelihood_function(v_array):
    parameters = {key: v for key, v in zip(sampling_keys, v_array)}
    if priors.evaluate_constraints(parameters):
        likelihood.parameters.update(parameters)
        return likelihood.log_likelihood() - likelihood.noise_log_likelihood()
    else:
        return np.nan_to_num(-np.inf)


sampling_keys = []
for p in priors:
    if isinstance(priors[p], bilby.core.prior.Constraint):
        continue
    if isinstance(priors[p], (int, float)):
        likelihood.parameters[p] = priors[p]
    elif priors[p].is_fixed:
        likelihood.parameters[p] = priors[p].peak
    else:
        sampling_keys.append(p)

# Setting marginalized parameters to their reference values
if likelihood.phase_marginalization:
    likelihood.parameters["phase"] = priors["phase"]
if likelihood.time_marginalization:
    likelihood.parameters["geocent_time"] = priors["geocent_time"]
if likelihood.distance_marginalization:
    likelihood.parameters["luminosity_distance"] = priors["luminosity_distance"]

t0 = datetime.datetime.now()
sampling_time = 0
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    POOL_SIZE = pool.size
    logger.info(f"sampling_keys={sampling_keys}")
    logger.info(priors)

    filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
    resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)

    sampler = NestedSampler(
        likelihood_function, prior_transform_function, len(sampling_keys),
        nlive=nlive, sample=input_args.dynesty_sample,
        walks=input_args.dynesty_walks, update_interval=int(2*nlive),
        pool=pool, queue_size=POOL_SIZE,
        print_func=dynesty.results.print_fn_fallback,
        use_pool=dict(update_bound=True,
                      propose_point=True,
                      prior_transform=True,
                      loglikelihood=True)
    )

    if os.path.isfile(resume_file):
        sampler, sampling_time = read_saved_state(resume_file, sampler)

    print(f"Starting sampling for job {label}, with pool size={POOL_SIZE}")
    old_ncall = sampler.ncall
    sampler_kwargs = dict(
        print_progress=True, maxcall=input_args.n_check_point,
        n_effective=input_args.n_effective, dlogz=input_args.dlogz)

    while True:
        sampler_kwargs['maxcall'] += input_args.n_check_point
        sampler.run_nested(**sampler_kwargs)
        if sampler.ncall == old_ncall:
            break
        old_ncall = sampler.ncall

        sampling_time += (datetime.datetime.now() - t0).total_seconds()
        t0 = datetime.datetime.now()
        write_checkpoint(sampler, resume_file, sampling_time, sampling_keys)

sampling_time += (datetime.datetime.now() - t0).total_seconds()

out = sampler.results
weights = np.exp(out['logwt'] - out['logz'][-1])
nested_samples = DataFrame(
    out.samples, columns=sampling_keys)
nested_samples['weights'] = weights
nested_samples['log_likelihood'] = out.logl

result = bilby.core.result.Result(
    label=label, outdir=outdir, search_parameter_keys=sampling_keys)
result.priors = priors
result.samples = dynesty.utils.resample_equal(out.samples, weights)
result.nested_samples = nested_samples
result.meta_data = dict()
result.meta_data["analysis_args"] = vars(input_args)
result.meta_data["config_file"] = vars(args)
result.meta_data["data_dump"] = input_args.data_dump
result.meta_data["likelihood"] = likelihood.meta_data
print(result.meta_data)

result.log_likelihood_evaluations = reorder_loglikelihoods(
    unsorted_loglikelihoods=out.logl, unsorted_samples=out.samples,
    sorted_samples=result.samples)

result.log_evidence = out.logz[-1]
result.log_evidence_err = out.logzerr[-1]
result.log_noise_evidence = likelihood.noise_log_likelihood()
result.sampling_time = sampling_time

result.samples_to_posterior()
result.save_to_file(extension="json")
