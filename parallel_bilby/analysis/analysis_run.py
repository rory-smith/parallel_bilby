import logging
import os
import pickle

import bilby
import dynesty
import numpy as np
from bilby.core.utils import logger

from .likelihood import setup_likelihood


class AnalysisRun(object):
    def __init__(self, input_args):

        # Get data_dump
        with open(input_args.data_dump, "rb") as file:
            data_dump = pickle.load(file)

        ifo_list = data_dump["ifo_list"]
        waveform_generator = data_dump["waveform_generator"]
        waveform_generator.start_time = ifo_list[0].time_array[0]
        args = data_dump["args"]
        injection_parameters = data_dump.get("injection_parameters", None)

        args.weight_file = data_dump["meta_data"].get("weight_file", None)

        outdir = args.outdir
        if input_args.outdir is not None:
            outdir = input_args.outdir
            os.makedirs(outdir, exist_ok=True)
        label = args.label
        if input_args.label is not None:
            label = input_args.label

        priors = bilby.gw.prior.PriorDict.from_json(data_dump["prior_file"])

        logger.setLevel(logging.WARNING)
        likelihood = setup_likelihood(
            interferometers=ifo_list,
            waveform_generator=waveform_generator,
            priors=priors,
            args=args,
        )
        priors.convert_floats_to_delta_functions()
        logger.setLevel(logging.INFO)

        sampling_keys = []
        for p in priors:
            if isinstance(priors[p], bilby.core.prior.Constraint):
                continue
            elif priors[p].is_fixed:
                likelihood.parameters[p] = priors[p].peak
            else:
                sampling_keys.append(p)

        periodic = []
        reflective = []
        for ii, key in enumerate(sampling_keys):
            if priors[key].boundary == "periodic":
                logger.debug(f"Setting periodic boundary for {key}")
                periodic.append(ii)
            elif priors[key].boundary == "reflective":
                logger.debug(f"Setting reflective boundary for {key}")
                reflective.append(ii)

        if input_args.dynesty_sample == "rwalk":
            logger.debug("Using the bilby-implemented rwalk sample method")
            dynesty.dynesty._SAMPLING[
                "rwalk"
            ] = bilby.core.sampler.dynesty.sample_rwalk_bilby
            dynesty.nestedsamplers._SAMPLING[
                "rwalk"
            ] = bilby.core.sampler.dynesty.sample_rwalk_bilby
        elif input_args.dynesty_sample == "rwalk_dynesty":
            logger.debug("Using the dynesty-implemented rwalk sample method")
            input_args.dynesty_sample = "rwalk"
        else:
            logger.debug(
                f"Using the dynesty-implemented {input_args.dynesty_sample} sample method"
            )

        self.init_sampler_kwargs = dict(
            nlive=input_args.nlive,
            sample=input_args.dynesty_sample,
            bound=input_args.dynesty_bound,
            walks=input_args.walks,
            maxmcmc=input_args.maxmcmc,
            nact=input_args.nact,
            facc=input_args.facc,
            first_update=dict(
                min_eff=input_args.min_eff, min_ncall=2 * input_args.nlive
            ),
            vol_dec=input_args.vol_dec,
            vol_check=input_args.vol_check,
            enlarge=input_args.enlarge,
            save_bounds=False,
        )

        self.sampling_seed = input_args.sampling_seed
        self.rstate = np.random.Generator(np.random.PCG64(self.sampling_seed))
        logger.debug(
            f"Setting random state = {self.rstate} (seed={self.sampling_seed})"
        )

        self.outdir = outdir
        self.label = label
        self.data_dump = data_dump
        self.priors = priors
        self.sampling_keys = sampling_keys
        self.likelihood = likelihood
        self.zero_likelihood_mode = input_args.bilby_zero_likelihood_mode
        self.periodic = periodic
        self.reflective = reflective
        self.args = args
        self.injection_parameters = injection_parameters
        self.nlive = input_args.nlive

    def prior_transform_function(self, u_array):
        return self.priors.rescale(self.sampling_keys, u_array)

    def log_likelihood_function(self, v_array):
        if self.zero_likelihood_mode:
            return 0
        parameters = {key: v for key, v in zip(self.sampling_keys, v_array)}
        if self.priors.evaluate_constraints(parameters) > 0:
            self.likelihood.parameters.update(parameters)
            return (
                self.likelihood.log_likelihood()
                - self.likelihood.noise_log_likelihood()
            )
        else:
            return np.nan_to_num(-np.inf)

    def log_prior_function(self, v_array):
        params = {key: t for key, t in zip(self.sampling_keys, v_array)}
        return self.priors.ln_prob(params)

    def get_initial_points_from_prior(self, pool, calculate_likelihood=True):

        # Create a new rstate for each point, otherwise each task will generate
        # the same random number, and the rstate on master will not be incremented
        sg = np.random.SeedSequence(self.rstate.integers(9223372036854775807))
        map_rstates = [
            np.random.Generator(np.random.PCG64(n)) for n in sg.spawn(self.nlive)
        ]
        ndim = len(self.sampling_keys)

        args_list = [
            (
                self.prior_transform_function,
                self.log_prior_function,
                self.log_likelihood_function,
                ndim,
                calculate_likelihood,
                map_rstates[i],
            )
            for i in range(self.nlive)
        ]
        initial_points = pool.map(self.get_initial_point_from_prior, args_list)
        u_list = [point[0] for point in initial_points]
        v_list = [point[1] for point in initial_points]
        l_list = [point[2] for point in initial_points]

        return np.array(u_list), np.array(v_list), np.array(l_list)

    @staticmethod
    def get_initial_point_from_prior(args):
        """
        Draw initial points from the prior subject to constraints applied both to
        the prior and the likelihood.

        We remove any points where the likelihood or prior is infinite or NaN.

        The `log_likelihood_function` often converts infinite values to large
        finite values so we catch those.
        """
        (
            prior_transform_function,
            log_prior_function,
            log_likelihood_function,
            ndim,
            calculate_likelihood,
            rstate,
        ) = args
        bad_values = [np.inf, np.nan_to_num(np.inf), np.nan]
        while True:
            unit = rstate.random(ndim)
            theta = prior_transform_function(unit)

            if abs(log_prior_function(theta)) not in bad_values:
                if calculate_likelihood:
                    logl = log_likelihood_function(theta)
                    if abs(logl) not in bad_values:
                        return unit, theta, logl
                else:
                    return unit, theta, np.nan
