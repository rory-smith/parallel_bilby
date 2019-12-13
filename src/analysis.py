#!/usr/bin/env python
"""
Script to run paralell bilby using MPI
"""
import os
import sys
import argparse
import datetime
import pickle
import logging
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bilby
from bilby.gw import conversion
import dynesty
from dynesty.plotting import traceplot
from dynesty import NestedSampler
from dynesty.utils import resample_equal, unitcheck
from schwimmbad import MPIPool
from pandas import DataFrame
from numpy import linalg

from bilby.core.utils import reflect

import mpi4py
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

logger = bilby.core.utils.logger


def main():
    """ Do nothing function to play nicely with MPI """
    pass


def fill_sample(args):
    ii, sample = args
    sample = dict(sample).copy()
    marg_params = likelihood.parameters.copy()
    likelihood.parameters.update(sample)
    sample = likelihood.generate_posterior_sample_from_marginalized_likelihood()
    # Likelihood needs to have marg params to calcuate correct SNR
    likelihood.parameters.update(marg_params)
    bilby.gw.conversion.compute_snrs(sample, likelihood)
    sample = conversion.generate_all_bbh_parameters(sample)
    return sample


def sample_rwalk_parallel_with_act(args):
    """ A dynesty sampling method optimised for parallel_bilby

    """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    # Bounds
    nonbounded = kwargs.get('nonbounded', None)
    periodic = kwargs.get('periodic', None)
    reflective = kwargs.get('reflective', None)

    # Setup.
    n = len(u)
    walks = kwargs.get('walks', 50)  # minimum number of steps
    maxmcmc = kwargs.get('maxmcmc', 10000)  # maximum number of steps
    nact = kwargs.get('nact', 10)  # number of act

    accept = 0
    reject = 0
    nfail = 0
    act = np.inf
    u_list = [u]
    v_list = [prior_transform(u)]
    logl_list = [loglikelihood(v_list[-1])]

    drhat, dr, du, u_prop, logl_prop = np.nan, np.nan, np.nan, np.nan, np.nan
    while len(u_list) < nact * act:

        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Scale based on dimensionality.
        # dr = drhat * rstate.rand()**(1./n)  # REMOVE
        dr = drhat * rstate.rand(n)

        # Transform to proposal distribution.
        du = np.dot(axes, dr)
        u_prop = u + scale * du

        # Wrap periodic parameters
        if periodic is not None:
            u_prop[periodic] = np.mod(u_prop[periodic], 1)
        # Reflect
        if reflective is not None:
            u_prop[reflective] = reflect(u_prop[reflective])

        # Check unit cube constraints.
        if unitcheck(u_prop, nonbounded):
            pass
        else:
            nfail += 1
            u_list.append(u_list[-1])
            v_list.append(v_list[-1])
            logl_list.append(logl_list[-1])
            continue

        # Check proposed point.
        v_prop = prior_transform(u_prop)
        logl_prop = loglikelihood(v_prop)
        if logl_prop >= loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
            u_list.append(u)
            v_list.append(v)
            logl_list.append(logl)
        else:
            reject += 1
            u_list.append(u_list[-1])
            v_list.append(v_list[-1])
            logl_list.append(logl_list[-1])

        # If we've taken the minimum number of steps, calculate the ACT
        if accept + reject > walks:
            act = estimate_nmcmc(
                accept / (accept + reject + nfail), walks, maxmcmc)

        # If we've taken too many likelihood evaluations then break
        if accept + reject > maxmcmc:
            logger.warning(
                "Hit maximum number of walks {} with accept={}, reject={}, "
                "nfail={}, and act={}. Try increasing maxmcmc"
                .format(maxmcmc, accept, reject, nfail, act))
            break

    # If the act is finite, pick randomly from within the chain
    if np.isfinite(act) and act < len(u_list):
        idx = np.random.randint(act, len(u_list))
        u = u_list[idx]
        v = v_list[idx]
        logl = logl_list[idx]
    elif len(u_list) == 1:
        logger.warning("Returning the only point in the chain")
        u = u_list[-1]
        v = v_list[-1]
        logl = logl_list[-1]
    else:
        idx = np.random.randint(int(len(u_list) / 2), len(u_list))
        logger.warning("Returning random point in second half of the chain")
        u = u_list[idx]
        v = v_list[idx]
        logl = logl_list[idx]

    blob = {'accept': accept, 'reject': reject, 'fail': nfail, 'scale': scale}

    ncall = accept + reject
    return u, v, logl, ncall, blob


def estimate_nmcmc(accept_ratio, minmcmc, maxmcmc, safety=5, tau=None):
    """ Estimate autocorrelation length of chain using acceptance fraction

    Using ACL = (2/acc) - 1 multiplied by a safety margin. Code adapated from
    CPNest:
        - https://github.com/johnveitch/cpnest/blob/master/cpnest/sampler.py
        - http://github.com/farr/Ensemble.jl

    Parameters
    ----------
    accept_ratio: float [0, 1]
        Ratio of the number of accepted points to the total number of points
    minmcmc: int
        The minimum length of the MCMC chain to use
    maxmcmc: int
        The maximum length of the MCMC chain to use
    safety: int
        A safety factor applied in the calculation
    tau: int (optional)
        The ACT, if given, otherwise estimated.

    """
    if tau is None:
        tau = maxmcmc / safety

    if accept_ratio == 0.0:
        Nmcmc_exact = (1. + 1. / tau) * minmcmc
    else:
        Nmcmc_exact = (
            (1. - 1. / tau) * minmcmc +
            (safety / tau) * (2. / accept_ratio - 1.)
        )

    Nmcmc_exact = float(min(Nmcmc_exact, maxmcmc))
    Nmcmc = max(safety, int(Nmcmc_exact))

    return Nmcmc


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
        sampler, resume_file, sampling_time, search_parameter_keys,
        no_plot=False):
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
    no_plot: bool
        If true, don't create a check point plot

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
    if no_plot is False:
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
    "-n", "--nlive", default=1024, type=int, help="Number of live points")
parser.add_argument(
    "--dlogz", default=0.1, type=float,
    help="Stopping criteria: remaining evidence, (default=0.1)")
parser.add_argument(
    "--n-effective", default=5000, type=float,
    help="Stopping criteria: effective number of samples, (default=5000)")
parser.add_argument(
    "--dynesty-sample", default="rwalk", type=str,
    help="Dynesty sampling method (default=rwalk). Note, the dynesty rwalk "
         "method is overwritten by parallel bilby for an optimised version ")
parser.add_argument(
    "--dynesty-bound", default="multi", type=str,
    help="Dynesty bounding method (default=multi)")
parser.add_argument(
    "--walks", default=100, type=int,
    help="Minimum number of walks, defaults to 100")
parser.add_argument(
    "--maxmcmc", default=5000, type=int,
    help="Maximum number of walks, defaults to 5000")
parser.add_argument(
    "--nact", default=5, type=int,
    help="Number of autocorrelation times to take, defaults to 5")
parser.add_argument(
    "--facc", default=0.5, type=float, help="See dynesty.NestedSampler")
parser.add_argument(
    "--vol-dec", default=0.5, type=float, help="See dynesty.NestedSampler")
parser.add_argument(
    "--vol-check", default=8, type=float, help="See dynesty.NestedSampler")
parser.add_argument(
    "--enlarge", default=1.5, type=float, help="See dynesty.NestedSampler")
parser.add_argument(
    "--n-check-point", default=10000, type=int,
    help="Number of walks")
parser.add_argument(
    "--bilby-zero-likelihood-mode", default=False, action="store_true")
parser.add_argument(
    "--rand-seed", type=int, default=1234,
    help="Random seed: important for reproducible resampling")
parser.add_argument(
    "-c", "--clean", action="store_true",
    help="Run clean: ignore any resume files")
parser.add_argument(
    "--no-plot", action="store_true",
    help="If true, don't generate check-point plots")


input_args = parser.parse_args()

with open(input_args.data_dump, "rb") as file:
    data_dump = pickle.load(file)

ifo_list = data_dump["ifo_list"]
waveform_generator = data_dump["waveform_generator"]
waveform_generator.start_time = ifo_list[0].time_array[0]
args = data_dump["args"]

outdir = args.outdir
label = args.label
if input_args.label is not None:
    label += "_" + "_".join(input_args.label)

if args.binary_neutron_star:
    priors = bilby.gw.prior.BNSPriorDict.from_json(data_dump["prior_file"])
else:
    priors = bilby.gw.prior.BBHPriorDict.from_json(data_dump["prior_file"])

logger.setLevel(logging.WARNING)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=priors,
    time_marginalization=args.time_marginalization,
    phase_marginalization=args.phase_marginalization,
    distance_marginalization=args.distance_marginalization,
    distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
    jitter_time=False)
logger.setLevel(logging.INFO)


def prior_transform_function(u_array):
    return priors.rescale(sampling_keys, u_array)


def likelihood_function(v_array):
    if input_args.bilby_zero_likelihood_mode:
        return 0
    parameters = {key: v for key, v in zip(sampling_keys, v_array)}
    if priors.evaluate_constraints(parameters) > 0:
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

periodic = []
reflective = []
for ii, key in enumerate(sampling_keys):
    if priors[key].boundary == 'periodic':
        logger.debug("Setting periodic boundary for {}".format(key))
        periodic.append(ii)
    elif priors[key].boundary == 'reflective':
        logger.debug("Setting reflective boundary for {}".format(key))
        reflective.append(ii)

# Setting marginalized parameters to their reference values
if likelihood.phase_marginalization:
    likelihood.parameters["phase"] = priors["phase"]
if likelihood.time_marginalization:
    likelihood.parameters["geocent_time"] = priors["geocent_time"]
if likelihood.distance_marginalization:
    likelihood.parameters["luminosity_distance"] = priors["luminosity_distance"]

if input_args.dynesty_sample == "rwalk":
    logger.debug(
        "Using the bilby-implemented rwalk sample method with estimated walks")
    dynesty.dynesty._SAMPLING["rwalk"] = sample_rwalk_parallel_with_act
    dynesty.nestedsamplers._SAMPLING["rwalk"] = sample_rwalk_parallel_with_act
elif input_args.dynesty_sample == "rwalk_dynesty":
    logger.debug(
        "Using the dynesty-implemented rwalk sample method")
    input_args.dynesty_sample = "rwalk"
else:
    logger.debug(
        "Using the dynesty-implemented {} sample method"
        .format(input_args.dynesty_sample))

t0 = datetime.datetime.now()
sampling_time = 0
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    POOL_SIZE = pool.size
    logger.info(f"sampling_keys={sampling_keys}")
    logger.info(
        "Periodic keys: {}".format([sampling_keys[ii] for ii in periodic]))
    logger.info(
        "Reflective keys: {}".format([sampling_keys[ii] for ii in reflective]))
    logger.info("Using priors:")
    for key in priors:
        logger.info(f"{key}: {priors[key]}")

    filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
    resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)

    dynesty_sample = input_args.dynesty_sample
    dynesty_bound = input_args.dynesty_bound
    nlive = input_args.nlive
    walks = input_args.walks
    maxmcmc = input_args.maxmcmc
    nact = input_args.nact
    facc = input_args.facc

    init_sampler_kwargs = dict(
        nlive=input_args.nlive, sample=input_args.dynesty_sample,
        bound=input_args.dynesty_bound, walks=input_args.walks,
        maxmcmc=input_args.maxmcmc, nact=input_args.nact, facc=input_args.facc,
        first_update=dict(min_eff=10, min_ncall=2*input_args.nlive),
        vol_dec=input_args.vol_dec, vol_check=input_args.vol_check,
        enlarge=input_args.enlarge)

    logger.info(
        "Initialize NestedSampler with {}"
        .format(json.dumps(init_sampler_kwargs, indent=1, sort_keys=True)))

    sampler = NestedSampler(
        likelihood_function, prior_transform_function, len(sampling_keys),
        pool=pool, queue_size=POOL_SIZE,
        print_func=dynesty.results.print_fn_fallback,
        periodic=periodic, reflective=reflective,
        use_pool=dict(update_bound=True,
                      propose_point=True,
                      prior_transform=True,
                      loglikelihood=True),
        **init_sampler_kwargs
    )

    if os.path.isfile(resume_file) and input_args.clean is False:
        resume_sampler, sampling_time = read_saved_state(resume_file, sampler)
        if resume_sampler is not False:
            sampler = resume_sampler

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
        write_checkpoint(sampler, resume_file, sampling_time, sampling_keys,
                         no_plot=input_args.no_plot)

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
    result.meta_data["command_line_args"] = vars(input_args)
    result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
    result.meta_data["config_file"] = vars(args)
    result.meta_data["data_dump"] = input_args.data_dump
    result.meta_data["likelihood"] = likelihood.meta_data
    result.meta_data["sampler_kwargs"] = init_sampler_kwargs
    result.meta_data["run_sampler_kwargs"] = sampler_kwargs

    result.log_likelihood_evaluations = reorder_loglikelihoods(
        unsorted_loglikelihoods=out.logl, unsorted_samples=out.samples,
        sorted_samples=result.samples)

    result.log_evidence = out.logz[-1] + likelihood.noise_log_likelihood()
    result.log_evidence_err = out.logzerr[-1]
    result.log_noise_evidence = likelihood.noise_log_likelihood()
    result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
    result.sampling_time = sampling_time

    result.samples_to_posterior()

    np.random.seed(input_args.rand_seed)
    posterior = result.posterior

    nsamples = len(posterior)
    logger.info("Using {} samples".format(nsamples))

    posterior = conversion.fill_from_fixed_priors(posterior, priors)

    logger.info("Generating posterior from marginalized parameters for"
                f" nsamples={len(posterior)}, POOL={pool.size}")
    samples = pool.map(fill_sample, posterior.iterrows())
    result.posterior = pd.DataFrame(samples)

    logger.debug("Updating prior to the actual prior")
    for par, name in zip(
            ['distance', 'phase', 'time'],
            ['luminosity_distance', 'phase', 'geocent_time']):
        if getattr(likelihood, '{}_marginalization'.format(par), False):
            priors[name] = likelihood.priors[name]
    result.priors = priors

    if args.convert_to_flat_in_component_mass:
        try:
            result = bilby.gw.prior.convert_to_flat_in_component_mass_prior(result)
        except Exception as e:
            logger.warning(f"Unable to convert to the LALInference prior: {e}")

    logger.info(f"Saving result to {outdir}/{label}_result.json")
    result.save_to_file(extension="json")
    print("Sampling time = {}s".format(datetime.timedelta(seconds=result.sampling_time)))
    print(result)
