#!/usr/bin/env python
"""
Script to run paralell bilby using MPI
"""
import os
import sys
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import bilby
import dynesty
from dynesty.plotting import traceplot
from dynesty import NestedSampler
from schwimmbad import MPIPool
from pandas import DataFrame

import mpi4py
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False


logger = bilby.core.utils.logger


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


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "0"
os.environ["MPI_PER_NODE"] = "16"

data_dump = sys.argv[1]
with open(data_dump, "rb") as file:
    data_dump = pickle.load(file)

priors = data_dump["priors"]
likelihood = data_dump["likelihood"]
args = data_dump["args"]
outdir = args.outdir
label = args.label
nlive = 1000


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
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    POOL_SIZE = pool.size
    logger.info(f"sampling_keys={sampling_keys}")
    logger.info(priors)
    sampler = NestedSampler(
        likelihood_function, prior_transform_function, len(sampling_keys),
        nlive=nlive, sample='unif', update_interval=4.1,
        pool=pool, queue_size=POOL_SIZE,
        use_pool=dict(update_bound=True,
                      propose_point=True,
                      prior_transform=True,
                      loglikelihood=True)
    )

    print(f"Starting sampling for job {label}, with pool size={POOL_SIZE}")
    old_ncall = sampler.ncall
    n_check_point = 10000
    sampler_kwargs = dict(
        print_progress=True, maxcall=n_check_point, n_effective=1000)
    filename = "{}/{}_checkpoint_trace.png".format(outdir, label)
    while True:
        sampler_kwargs['maxcall'] += n_check_point
        sampler.run_nested(**sampler_kwargs)
        if sampler.ncall == old_ncall:
            break
        old_ncall = sampler.ncall

        try:
            fig = traceplot(sampler.results, labels=sampling_keys)[0]
            fig.tight_layout()
            fig.savefig(filename)
            plt.close('all')
        except Exception:
            pass
sampling_time = (datetime.datetime.now() - t0).total_seconds()

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

result.log_likelihood_evaluations = reorder_loglikelihoods(
    unsorted_loglikelihoods=out.logl, unsorted_samples=out.samples,
    sorted_samples=result.samples)

result.log_evidence = out.logz[-1]
result.log_evidence_err = out.logzerr[-1]
result.log_noise_evidence = likelihood.noise_log_likelihood()
result.sampling_time = sampling_time

result.samples_to_posterior()
result.save_to_file(extension="json")
