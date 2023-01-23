"""
Module to run parallel bilby using MPI
"""
import datetime
import json
import os
import pickle
import time

import bilby
import numpy as np
import pandas as pd
from bilby.core.utils import logger
from bilby.gw import conversion
from nestcheck import data_processing
from pandas import DataFrame

from ..parser import create_analysis_parser
from ..schwimmbad_fast import MPIPoolFast as MPIPool
from ..utils import get_cli_args, stdout_sampling_log
from .analysis_run import AnalysisRun
from .plotting import plot_current_state
from .read_write import (
    format_result,
    read_saved_state,
    write_current_state,
    write_sample_dump,
)
from .sample_space import fill_sample


def analysis_runner(
    data_dump,
    outdir=None,
    label=None,
    dynesty_sample="rwalk",
    nlive=5,
    dynesty_bound="multi",
    walks=100,
    maxmcmc=5000,
    nact=1,
    facc=0.5,
    min_eff=10,
    vol_dec=0.5,
    vol_check=8,
    enlarge=1.5,
    sampling_seed=0,
    bilby_zero_likelihood_mode=False,
    #
    fast_mpi=False,
    mpi_timing=False,
    mpi_timing_interval=0,
    check_point_deltaT=3600,
    n_effective=np.inf,
    dlogz=10,
    do_not_save_bounds_in_resume=False,
    n_check_point=1e4,
    max_its=1e10,
    max_run_time=1e10,
    rotate_checkpoints=False,
    no_plot=False,
    nestcheck=False,
    **kwargs,
):
    """
    API for running the analysis from Python instead of the command line.
    It takes all the same options as the CLI, specified as keyword arguments.

    Returns
    -------
    exit_reason: integer u
        Used during testing, to determine the reason the code halted:
            0 = run completed normally, based on convergence criteria
            1 = reached max iterations
            2 = reached max runtime
        MPI worker tasks always return -1

    """

    # Initialise a run
    run = AnalysisRun(
        data_dump,
        outdir,
        label,
        dynesty_sample,
        nlive,
        dynesty_bound,
        walks,
        maxmcmc,
        nact,
        facc,
        min_eff,
        vol_dec,
        vol_check,
        enlarge,
        sampling_seed,
        bilby_zero_likelihood_mode,
    )

    t0 = datetime.datetime.now()
    sampling_time = 0
    with MPIPool(
        parallel_comms=fast_mpi,
        time_mpi=mpi_timing,
        timing_interval=mpi_timing_interval,
        use_dill=True,
    ) as pool:
        if pool.is_master():
            POOL_SIZE = pool.size

            logger.info(f"sampling_keys={run.sampling_keys}")
            if run.periodic:
                logger.info(
                    f"Periodic keys: {[run.sampling_keys[ii] for ii in run.periodic]}"
                )
            if run.reflective:
                logger.info(
                    f"Reflective keys: {[run.sampling_keys[ii] for ii in run.reflective]}"
                )
            logger.info("Using priors:")
            for key in run.priors:
                logger.info(f"{key}: {run.priors[key]}")

            resume_file = f"{run.outdir}/{run.label}_checkpoint_resume.pickle"
            samples_file = f"{run.outdir}/{run.label}_samples.dat"

            sampler, sampling_time = read_saved_state(resume_file)

            if sampler is False:
                logger.info(f"Initializing sampling points with pool size={POOL_SIZE}")
                live_points = run.get_initial_points_from_prior(pool)
                logger.info(
                    f"Initialize NestedSampler with "
                    f"{json.dumps(run.init_sampler_kwargs, indent=1, sort_keys=True)}"
                )
                sampler = run.get_nested_sampler(live_points, pool, POOL_SIZE)
            else:
                # Reinstate the pool and map (not saved in the pickle)
                logger.info(f"Read in resume file with sampling_time = {sampling_time}")
                sampler.pool = pool
                sampler.M = pool.map
                sampler.loglikelihood.pool = pool

            logger.info(
                f"Starting sampling for job {run.label}, with pool size={POOL_SIZE} "
                f"and check_point_deltaT={check_point_deltaT}"
            )

            sampler_kwargs = dict(
                n_effective=n_effective,
                dlogz=dlogz,
                save_bounds=not do_not_save_bounds_in_resume,
            )
            logger.info(f"Run criteria: {json.dumps(sampler_kwargs)}")

            run_time = 0
            early_stop = False

            for it, res in enumerate(sampler.sample(**sampler_kwargs)):
                stdout_sampling_log(
                    results=res, niter=it, ncall=sampler.ncall, dlogz=dlogz
                )

                iteration_time = (datetime.datetime.now() - t0).total_seconds()
                t0 = datetime.datetime.now()

                sampling_time += iteration_time
                run_time += iteration_time

                if os.path.isfile(resume_file):
                    last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
                else:
                    last_checkpoint_s = np.inf

                """
                Criteria for writing checkpoints:
                a) time since last checkpoint > check_point_deltaT
                b) reached an integer multiple of n_check_point
                c) reached max iterations
                d) reached max runtime
                """

                if (
                    last_checkpoint_s > check_point_deltaT
                    or (it % n_check_point == 0 and it != 0)
                    or it == max_its
                    or run_time > max_run_time
                ):

                    write_current_state(
                        sampler,
                        resume_file,
                        sampling_time,
                        rotate_checkpoints,
                    )
                    write_sample_dump(sampler, samples_file, run.sampling_keys)
                    if no_plot is False:
                        plot_current_state(
                            sampler, run.sampling_keys, run.outdir, run.label
                        )

                    if it == max_its:
                        exit_reason = 1
                        logger.info(
                            f"Max iterations ({it}) reached; stopping sampling (exit_reason={exit_reason})."
                        )
                        early_stop = True
                        break

                    if run_time > max_run_time:
                        exit_reason = 2
                        logger.info(
                            f"Max run time ({max_run_time}) reached; stopping sampling (exit_reason={exit_reason})."
                        )
                        early_stop = True
                        break

            if not early_stop:
                exit_reason = 0
                # Adding the final set of live points.
                for it_final, res in enumerate(sampler.add_live_points()):
                    pass

                # Create a final checkpoint and set of plots
                write_current_state(
                    sampler, resume_file, sampling_time, rotate_checkpoints
                )
                write_sample_dump(sampler, samples_file, run.sampling_keys)
                if no_plot is False:
                    plot_current_state(
                        sampler, run.sampling_keys, run.outdir, run.label
                    )

                sampling_time += (datetime.datetime.now() - t0).total_seconds()

                out = sampler.results

                if nestcheck is True:
                    logger.info("Creating nestcheck files")
                    ns_run = data_processing.process_dynesty_run(out)
                    nestcheck_path = os.path.join(run.outdir, "Nestcheck")
                    bilby.core.utils.check_directory_exists_and_if_not_mkdir(
                        nestcheck_path
                    )
                    nestcheck_result = f"{nestcheck_path}/{run.label}_nestcheck.pickle"

                    with open(nestcheck_result, "wb") as file_nest:
                        pickle.dump(ns_run, file_nest)

                weights = np.exp(out["logwt"] - out["logz"][-1])
                nested_samples = DataFrame(out.samples, columns=run.sampling_keys)
                nested_samples["weights"] = weights
                nested_samples["log_likelihood"] = out.logl

                result = format_result(
                    run,
                    data_dump,
                    out,
                    weights,
                    nested_samples,
                    sampler_kwargs,
                    sampling_time,
                )

                posterior = conversion.fill_from_fixed_priors(
                    result.posterior, run.priors
                )

                logger.info(
                    "Generating posterior from marginalized parameters for"
                    f" nsamples={len(posterior)}, POOL={pool.size}"
                )
                fill_args = [
                    (ii, row, run.likelihood) for ii, row in posterior.iterrows()
                ]
                samples = pool.map(fill_sample, fill_args)
                result.posterior = pd.DataFrame(samples)

                logger.debug("Updating prior to the actual prior")
                for par, name in zip(
                    ["distance", "phase", "time"],
                    ["luminosity_distance", "phase", "geocent_time"],
                ):
                    if getattr(run.likelihood, f"{par}_marginalization", False):
                        run.priors[name] = run.likelihood.priors[name]
                result.priors = run.priors

                logger.info(f"Saving result to {run.outdir}/{run.label}_result.json")
                result.save_to_file(extension="json")
                print(
                    f"Sampling time = {datetime.timedelta(seconds=result.sampling_time)}s"
                )
                print(result)
        else:
            exit_reason = -1
        return exit_reason


def main():
    """
    paralell_bilby_analysis entrypoint.

    This function is a wrapper around analysis_runner(),
    giving it a command line interface.
    """
    cli_args = get_cli_args()

    # Parse command line arguments
    analysis_parser = create_analysis_parser(sampler="dynesty")
    input_args = analysis_parser.parse_args(args=cli_args)

    # Run the analysis
    analysis_runner(**vars(input_args))