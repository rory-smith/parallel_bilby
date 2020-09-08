#!/usr/bin/env python
"""
Module to run parallel bilby using MPI and ptemcee
"""
import datetime
import logging
import os
import pickle
import sys
import time

import bilby
import dill
import mpi4py
import numpy as np
import pandas as pd
import ptemcee
import tqdm
from bilby.core.sampler.ptemcee import (
    ConvergenceInputs,
    check_iteration,
    checkpoint,
    compute_evidence,
    plot_tau,
    plot_walkers,
)
from bilby.gw import conversion
from schwimmbad import MPIPool

from .parser import create_analysis_parser
from .utils import fill_sample, get_initial_points_from_prior

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

logger = bilby.core.utils.logger


def main():
    """ Do nothing function to play nicely with MPI """
    pass


def get_pos0():
    p0_list = []
    for i in tqdm.tqdm(range(input_args.ntemps)):
        _, p0, _ = get_initial_points_from_prior(
            ndim,
            input_args.nwalkers,
            prior_transform_function,
            log_likelihood_function,
            pool,
            calculate_likelihood=False,
        )
        p0_list.append(p0)
    return np.array(p0_list)


def get_zero_chain_array():
    np.zeros((input_args.nwalkers, max_steps, ndim))


def get_zero_log_likelihood_array():
    np.zeros((input_args.ntemps, input_args.nwalkers, max_steps))


def write_current_state(plot=True):
    checkpoint(
        iteration,
        outdir,
        label,
        nsamples_effective,
        sampler,
        nburn,
        thin,
        search_parameter_keys,
        resume_file,
        log_likelihood_array,
        chain_array,
        pos0,
        beta_list,
        tau_list,
        tau_list_n,
        time_per_check,
    )

    if plot and not np.isnan(nburn):
        # Generate the walkers plot diagnostic
        plot_walkers(
            chain_array[:, :iteration, :],
            nburn,
            thin,
            search_parameter_keys,
            outdir,
            label,
        )

        # Generate the tau plot diagnostic
        plot_tau(
            tau_list_n,
            tau_list,
            search_parameter_keys,
            outdir,
            label,
            tau_int,
            convergence_inputs.autocorr_tau,
        )


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "0"
os.environ["MPI_PER_NODE"] = "16"

analysis_parser = create_analysis_parser(sampler="ptemcee")
input_args = analysis_parser.parse_args()

with open(input_args.data_dump, "rb") as file:
    data_dump = pickle.load(file)

# Load in data from the dump
ifo_list = data_dump["ifo_list"]
waveform_generator = data_dump["waveform_generator"]
waveform_generator.start_time = ifo_list[0].time_array[0]
args = data_dump["args"]
injection_parameters = data_dump.get("injection_parameters", None)
priors = bilby.gw.prior.PriorDict.from_json(data_dump["prior_file"])

# Overwrite outdir if given at the command line
outdir = args.outdir
if input_args.outdir is not None:
    outdir = input_args.outdir

# Overwrite label if given at the command line
label = args.label
if input_args.label is not None:
    label = input_args.label


logger.setLevel(logging.WARNING)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=args.time_marginalization,
    phase_marginalization=args.phase_marginalization,
    distance_marginalization=args.distance_marginalization,
    distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
    jitter_time=args.jitter_time,
)
logger.setLevel(logging.INFO)


def prior_transform_function(u_array):
    return priors.rescale(search_parameter_keys, u_array)


def log_likelihood_function(v_array):
    if input_args.bilby_zero_likelihood_mode:
        return 0

    log_prior = log_prior_function(v_array)
    if np.isinf(log_prior):
        return log_prior

    parameters = {key: v for key, v in zip(search_parameter_keys, v_array)}
    if priors.evaluate_constraints(parameters) > 0:
        likelihood.parameters.update(parameters)
        return likelihood.log_likelihood()
    else:
        return np.nan_to_num(-np.inf)


def log_prior_function(v_array):
    params = {key: t for key, t in zip(search_parameter_keys, v_array)}
    return priors.ln_prob(params)


search_parameter_keys = []
for p in priors:
    if isinstance(priors[p], bilby.core.prior.Constraint):
        continue
    if isinstance(priors[p], (int, float)):
        likelihood.parameters[p] = priors[p]
    elif priors[p].is_fixed:
        likelihood.parameters[p] = priors[p].peak
    else:
        search_parameter_keys.append(p)

# Setting marginalized parameters to their reference values
if likelihood.phase_marginalization:
    likelihood.parameters["phase"] = priors["phase"]
if likelihood.time_marginalization:
    likelihood.parameters["geocent_time"] = priors["geocent_time"]
if likelihood.distance_marginalization:
    likelihood.parameters["luminosity_distance"] = priors["luminosity_distance"]

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    POOL_SIZE = pool.size

    max_steps = 500

    logger.info("Setting sampling seed = {}".format(input_args.sampling_seed))
    np.random.seed(input_args.sampling_seed)

    logger.info("Using priors:")
    for key in priors:
        logger.info(f"{key}: {priors[key]}")

    ndim = len(search_parameter_keys)
    sampler_init_kwargs = dict(
        nwalkers=input_args.nwalkers,
        dim=ndim,
        ntemps=input_args.ntemps,
        Tmax=input_args.Tmax,
    )

    # Store convergence checking inputs in a named tuple
    convergence_inputs_dict = dict(
        autocorr_c=input_args.autocorr_c,
        autocorr_tol=input_args.autocorr_tol,
        autocorr_tau=input_args.autocorr_tau,
        safety=input_args.safety,
        burn_in_nact=input_args.burn_in_nact,
        thin_by_nact=input_args.thin_by_nact,
        frac_threshold=input_args.frac_threshold,
        nsamples=input_args.nsamples,
        ignore_keys_for_tau=None,
        min_tau=input_args.min_tau,
        niterations_per_check=input_args.niterations_per_check,
    )
    convergence_inputs = ConvergenceInputs(**convergence_inputs_dict)

    resume = True
    check_point_deltaT = args.check_point_deltaT
    check_point_plot = args.check_point_plot
    resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)

    if os.path.isfile(resume_file) and resume is True:
        logger.info("Resume data {} found".format(resume_file))
        with open(resume_file, "rb") as file:
            data = dill.load(file)

        # Extract the check-point data
        sampler = data["sampler"]
        iteration = data["iteration"]
        chain_array = data["chain_array"]
        log_likelihood_array = data["log_likelihood_array"]
        pos0 = data["pos0"]
        beta_list = data["beta_list"]
        sampler._betas = np.array(beta_list[-1])
        tau_list = data["tau_list"]
        tau_list_n = data["tau_list_n"]
        time_per_check = data["time_per_check"]

        # Initialize the pool
        sampler.pool = pool
        sampler.threads = POOL_SIZE

        logger.info("Resuming from previous run with time={}".format(iteration))

    else:
        # Initialize the PTSampler
        sampler = ptemcee.Sampler(
            dim=ndim,
            logl=log_likelihood_function,
            logp=log_prior_function,
            pool=pool,
            threads=POOL_SIZE,
            **sampler_init_kwargs,
        )

        # Initialize storing results
        iteration = 0
        chain_array = get_zero_chain_array()
        log_likelihood_array = get_zero_log_likelihood_array()
        beta_list = []
        tau_list = []
        tau_list_n = []
        time_per_check = []
        pos0 = get_pos0()

    logger.info(
        "Starting sampling: nsamples={}, burn_in_nact={}, thin_by_nact={},"
        " adapt={}, autocorr_c={}, autocorr_tol={}, ncheck={}".format(
            input_args.nsamples,
            input_args.burn_in_nact,
            input_args.thin_by_nact,
            input_args.adapt,
            input_args.autocorr_c,
            input_args.autocorr_tol,
            input_args.ncheck,
        )
    )

    t0 = datetime.datetime.now()
    time_per_check = []

    evals_per_check = input_args.nwalkers * input_args.ntemps * input_args.ncheck

    while True:
        for (pos0, log_posterior, log_likelihood) in sampler.sample(
            pos0,
            storechain=False,
            iterations=convergence_inputs.niterations_per_check,
            adapt=input_args.adapt,
        ):
            pass

        if iteration == chain_array.shape[1]:
            chain_array = np.concatenate((chain_array, get_zero_chain_array()), axis=1)
            log_likelihood_array = np.concatenate(
                (log_likelihood_array, get_zero_log_likelihood_array()), axis=2
            )

        pos0 = pos0
        chain_array[:, iteration, :] = pos0[0, :, :]
        log_likelihood_array[:, :, iteration] = log_likelihood

        # Calculate time per iteration
        time_per_check.append((datetime.datetime.now() - t0).total_seconds())
        t0 = datetime.datetime.now()

        iteration += 1

        (stop, nburn, thin, tau_int, nsamples_effective) = check_iteration(
            chain_array[:, : iteration + 1, :],
            sampler,
            convergence_inputs,
            search_parameter_keys,
            time_per_check,
            beta_list,
            tau_list,
            tau_list_n,
        )

        if stop:
            logger.info("Finished sampling")
            break

        # If a checkpoint is due, checkpoint
        if os.path.isfile(resume_file):
            last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
        else:
            last_checkpoint_s = np.sum(time_per_check)

        if last_checkpoint_s > check_point_deltaT:
            write_current_state(plot=check_point_plot)

    # Run a final checkpoint to update the plots and samples
    write_current_state(plot=check_point_plot)

    # Set up an empty result object
    result = bilby.core.result.Result(
        label=label, outdir=outdir, search_parameter_keys=search_parameter_keys
    )
    result.priors = priors
    result.nburn = nburn

    # Get 0-likelihood samples and store in the result
    result.samples = chain_array[:, nburn:iteration:thin, :].reshape((-1, ndim))
    loglikelihood = log_likelihood_array[0, :, nburn:iteration:thin]  # nwalkers, nsteps
    result.log_likelihood_evaluations = loglikelihood.reshape((-1))

    result.nburn = nburn

    log_evidence, log_evidence_err = compute_evidence(
        sampler, log_likelihood_array, outdir, label, nburn, thin, iteration
    )
    result.log_evidence = log_evidence
    result.log_evidence_err = log_evidence_err

    result.sampling_time = datetime.timedelta(seconds=np.sum(time_per_check))

    # Create and store the meta data and injection_parameters
    result.meta_data = data_dump["meta_data"]
    result.meta_data["command_line_args"] = vars(input_args)
    result.meta_data["command_line_args"]["sampler"] = "parallel_bilby_ptemcee"
    result.meta_data["config_file"] = vars(args)
    result.meta_data["data_dump"] = input_args.data_dump
    result.meta_data["sampler_kwargs"] = sampler_init_kwargs
    result.meta_data["likelihood"] = likelihood.meta_data
    result.meta_data["injection_parameters"] = injection_parameters
    result.injection_parameters = injection_parameters

    # Post-process the posterior
    posterior = result.posterior
    nsamples = len(posterior)
    logger.info("Using {} samples".format(nsamples))
    posterior = conversion.fill_from_fixed_priors(posterior, priors)
    logger.info(
        "Generating posterior from marginalized parameters for"
        f" nsamples={len(posterior)}, POOL={pool.size}"
    )
    fill_args = [(ii, row, likelihood) for ii, row in posterior.iterrows()]
    samples = pool.map(fill_sample, fill_args)

    result.posterior = pd.DataFrame(samples)

    logger.debug("Updating prior to the actual prior")
    for par, name in zip(
        ["distance", "phase", "time"], ["luminosity_distance", "phase", "geocent_time"]
    ):
        if getattr(likelihood, "{}_marginalization".format(par), False):
            priors[name] = likelihood.priors[name]
    result.priors = priors

    if args.convert_to_flat_in_component_mass:
        try:
            result = bilby.gw.prior.convert_to_flat_in_component_mass_prior(result)
        except Exception as e:
            logger.warning(f"Unable to convert to the LALInference prior: {e}")

    logger.info(f"Saving result to {outdir}/{label}_result.json")
    result.save_to_file(extension="json")
    print(
        "Sampling time = {}s".format(datetime.timedelta(seconds=result.sampling_time))
    )
    print(result)
