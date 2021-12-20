#!/usr/bin/env python
"""
Module to run parallel bilby using MPI
"""
import datetime
import json
import os
import pickle
import sys
import time

import bilby
import dynesty
import nestcheck.data_processing
import numpy as np
import pandas as pd
from bilby.core.utils import logger
from bilby.gw import conversion
from dynesty import NestedSampler
from pandas import DataFrame

from ..parser import create_analysis_parser
from ..schwimmbad_fast import MPIPoolFast as MPIPool
from ..utils import fill_sample, get_cli_args, get_initial_points_from_prior
from .plotting import plot_current_state
from .read_write import (
    format_result,
    read_saved_state,
    write_current_state,
    write_sample_dump,
)
from .run import AnalysisRun


def analysis_runner(cli_args):

    # Parse command line arguments
    analysis_parser = create_analysis_parser(sampler="dynesty")
    input_args = analysis_parser.parse_args(args=cli_args)

    run = AnalysisRun(input_args)

    outdir = run.outdir
    label = run.label
    data_dump = run.data_dump
    priors = run.priors
    sampling_keys = run.sampling_keys
    likelihood = run.likelihood
    periodic = run.periodic
    reflective = run.reflective
    args = run.args
    injection_parameters = run.injection_parameters
    init_sampler_kwargs = run.init_sampler_kwargs

    t0 = datetime.datetime.now()
    sampling_time = 0
    with MPIPool(
        parallel_comms=input_args.fast_mpi,
        time_mpi=input_args.mpi_timing,
        timing_interval=input_args.mpi_timing_interval,
    ) as pool:
        if pool.is_master():
            POOL_SIZE = pool.size

            sampling_seed = input_args.sampling_seed
            rstate = np.random.Generator(np.random.PCG64(sampling_seed))
            logger.info(f"Setting random state = {rstate} (seed={sampling_seed})")

            logger.info(f"sampling_keys={sampling_keys}")
            logger.info(f"Periodic keys: {[sampling_keys[ii] for ii in periodic]}")
            logger.info(f"Reflective keys: {[sampling_keys[ii] for ii in reflective]}")
            logger.info("Using priors:")
            for key in priors:
                logger.info(f"{key}: {priors[key]}")

            resume_file = f"{outdir}/{label}_checkpoint_resume.pickle"
            samples_file = f"{outdir}/{label}_samples.dat"

            ndim = len(sampling_keys)
            sampler, sampling_time = read_saved_state(resume_file)

            if sampler is False:
                logger.info(f"Initializing sampling points with pool size={POOL_SIZE}")
                live_points = get_initial_points_from_prior(
                    ndim,
                    input_args.nlive,
                    run.prior_transform_function,
                    run.log_prior_function,
                    run.log_likelihood_function,
                    pool,
                    rstate,
                )
                logger.info(
                    f"Initialize NestedSampler with "
                    f"{json.dumps(init_sampler_kwargs, indent=1, sort_keys=True)}"
                )
                sampler = NestedSampler(
                    run.log_likelihood_function,
                    run.prior_transform_function,
                    ndim,
                    pool=pool,
                    queue_size=POOL_SIZE,
                    print_func=dynesty.results.print_fn_fallback,
                    periodic=periodic,
                    reflective=reflective,
                    live_points=live_points,
                    rstate=rstate,
                    use_pool=dict(
                        update_bound=True,
                        propose_point=True,
                        prior_transform=True,
                        loglikelihood=True,
                    ),
                    **init_sampler_kwargs,
                )
            else:
                # Reinstate the pool and map (not saved in the pickle)
                logger.info(f"Read in resume file with sampling_time = {sampling_time}")
                sampler.pool = pool
                sampler.M = pool.map

            logger.info(
                f"Starting sampling for job {label}, with pool size={POOL_SIZE} "
                f"and check_point_deltaT={input_args.check_point_deltaT}"
            )

            sampler_kwargs = dict(
                n_effective=input_args.n_effective,
                dlogz=input_args.dlogz,
                save_bounds=not input_args.do_not_save_bounds_in_resume,
            )
            logger.info(f"Run criteria: {json.dumps(sampler_kwargs)}")

            run_time = 0

            for it, res in enumerate(sampler.sample(**sampler_kwargs)):
                i = it - 1
                dynesty.results.print_fn_fallback(
                    res, i, sampler.ncall, dlogz=input_args.dlogz
                )

                if (
                    it == 0 or it % input_args.n_check_point != 0
                ) and it != input_args.max_its:
                    continue

                iteration_time = (datetime.datetime.now() - t0).total_seconds()
                t0 = datetime.datetime.now()

                sampling_time += iteration_time
                run_time += iteration_time

                if os.path.isfile(resume_file):
                    last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
                else:
                    last_checkpoint_s = np.inf

                if (
                    last_checkpoint_s > input_args.check_point_deltaT
                    or it == input_args.max_its
                    or run_time > input_args.max_run_time
                ):
                    write_current_state(
                        sampler,
                        resume_file,
                        sampling_time,
                        input_args.rotate_checkpoints,
                    )
                    write_sample_dump(sampler, samples_file, sampling_keys)
                    if input_args.no_plot is False:
                        plot_current_state(sampler, sampling_keys, outdir, label)

                    if it == input_args.max_its:
                        logger.info(
                            f"Max iterations ({it}) reached; stopping sampling."
                        )
                        sys.exit(0)

                    if run_time > input_args.max_run_time:
                        logger.info(
                            f"Max run time ({input_args.max_run_time}) reached; stopping sampling."
                        )
                        sys.exit(0)

            # Adding the final set of live points.
            for it_final, res in enumerate(sampler.add_live_points()):
                pass

            # Create a final checkpoint and set of plots
            write_current_state(
                sampler, resume_file, sampling_time, input_args.rotate_checkpoints
            )
            write_sample_dump(sampler, samples_file, sampling_keys)
            if input_args.no_plot is False:
                plot_current_state(sampler, sampling_keys, outdir, label)

            sampling_time += (datetime.datetime.now() - t0).total_seconds()

            out = sampler.results

            if input_args.nestcheck is True:
                logger.info("Creating nestcheck files")
                ns_run = nestcheck.data_processing.process_dynesty_run(out)
                nestcheck_path = os.path.join(outdir, "Nestcheck")
                bilby.core.utils.check_directory_exists_and_if_not_mkdir(nestcheck_path)
                nestcheck_result = f"{nestcheck_path}/{label}_nestcheck.pickle"

                with open(nestcheck_result, "wb") as file_nest:
                    pickle.dump(ns_run, file_nest)

            weights = np.exp(out["logwt"] - out["logz"][-1])
            nested_samples = DataFrame(out.samples, columns=sampling_keys)
            nested_samples["weights"] = weights
            nested_samples["log_likelihood"] = out.logl

            result = format_result(
                label,
                outdir,
                sampling_keys,
                priors,
                out,
                weights,
                nested_samples,
                data_dump,
                input_args,
                args,
                likelihood,
                init_sampler_kwargs,
                sampler_kwargs,
                injection_parameters,
                sampling_time,
            )

            posterior = result.posterior

            nsamples = len(posterior)
            logger.info(f"Using {nsamples} samples")

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
                ["distance", "phase", "time"],
                ["luminosity_distance", "phase", "geocent_time"],
            ):
                if getattr(likelihood, f"{par}_marginalization", False):
                    priors[name] = likelihood.priors[name]
            result.priors = priors

            if args.convert_to_flat_in_component_mass:
                try:
                    result = bilby.gw.prior.convert_to_flat_in_component_mass_prior(
                        result
                    )
                except Exception as e:
                    logger.warning(f"Unable to convert to the LALInference prior: {e}")

            logger.info(f"Saving result to {outdir}/{label}_result.json")
            result.save_to_file(extension="json")
            print(
                f"Sampling time = {datetime.timedelta(seconds=result.sampling_time)}s"
            )
            print(result)


def main():
    cli_args = get_cli_args()
    analysis_runner(cli_args)
