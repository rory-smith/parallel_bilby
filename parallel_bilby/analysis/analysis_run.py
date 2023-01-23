import logging
import os
import pickle

import bilby
import dynesty
import numpy as np
from bilby.core.utils import logger

from .likelihood import setup_likelihood


class AnalysisRun(object):
    """
    An object with methods for driving the sampling run.

    Parameters: arguments to set the output path and control the dynesty sampler.
    """

    def __init__(
        self,
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
    ):

        # Read data dump from the pickle file
        with open(data_dump, "rb") as file:
            data_dump = pickle.load(file)

        ifo_list = data_dump["ifo_list"]
        waveform_generator = data_dump["waveform_generator"]
        waveform_generator.start_time = ifo_list[0].time_array[0]
        args = data_dump["args"]
        injection_parameters = data_dump.get("injection_parameters", None)

        args.weight_file = data_dump["meta_data"].get("weight_file", None)

        # If the run dir has not been specified, get it from the args
        if outdir is None:
            outdir = args.outdir
        else:
            # Create the run dir
            os.makedirs(outdir, exist_ok=True)

        # If the label has not been specified, get it from the args
        if label is None:
            label = args.label

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

        if len(periodic) == 0:
            periodic = None
        if len(reflective) == 0:
            reflective = None

        if dynesty_sample == "rwalk":
            logger.debug("Using the bilby-implemented rwalk sample method")
            dynesty.dynesty._SAMPLING[
                "rwalk"
            ] = bilby.core.sampler.dynesty.sample_rwalk_bilby
            dynesty.nestedsamplers._SAMPLING[
                "rwalk"
            ] = bilby.core.sampler.dynesty.sample_rwalk_bilby
        elif dynesty_sample == "rwalk_dynesty":
            logger.debug("Using the dynesty-implemented rwalk sample method")
            dynesty_sample = "rwalk"
        else:
            logger.debug(
                f"Using the dynesty-implemented {dynesty_sample} sample method"
            )

        self.init_sampler_kwargs = dict(
            nlive=nlive,
            sample=dynesty_sample,
            bound=dynesty_bound,
            walks=walks,
            # maxmcmc=maxmcmc,
            # nact=nact,
            facc=facc,
            first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
            # vol_dec=vol_dec,
            # vol_check=vol_check,
            enlarge=enlarge,
            # save_bounds=False,
        )

        # Create a random generator, which is saved across restarts
        # This ensures that runs are fully deterministic, which is important
        # for testing
        self.sampling_seed = sampling_seed
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
        self.zero_likelihood_mode = bilby_zero_likelihood_mode
        self.periodic = periodic
        self.reflective = reflective
        self.args = args
        self.injection_parameters = injection_parameters
        self.nlive = nlive

    def prior_transform_function(self, u_array):
        """
        Calls the bilby rescaling function on an array of values

        Parameters
        ----------
        u_array: (float, array-like)
            The values to rescale

        Returns
        -------
        (float, array-like)
            The rescaled values

        """
        return self.priors.rescale(self.sampling_keys, u_array)

    def log_likelihood_function(self, v_array):
        """
        Calculates the log(likelihood)

        Parameters
        ----------
        u_array: (float, array-like)
            The values to rescale

        Returns
        -------
        (float, array-like)
            The rescaled values

        """
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
        """
        Calculates the log of the prior

        Parameters
        ----------
        v_array: (float, array-like)
            The prior values

        Returns
        -------
        (float, array-like)
            The log probability of the values

        """
        params = {key: t for key, t in zip(self.sampling_keys, v_array)}
        return self.priors.ln_prob(params)

    def get_initial_points_from_prior(self, pool, calculate_likelihood=True):
        """
        Generates a set of initial points, drawn from the prior

        Parameters
        ----------
        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation
            (pbilby implements a modified version: MPIPoolFast)

        calculate_likelihood: bool
            Option to calculate the likelihood for the generated points
            (default: True)

        Returns
        -------
        (numpy.ndarraym, numpy.ndarray, numpy.ndarray, None)
            Returns a tuple (unit, theta, logl, blob)
            unit: point in the unit cube
            theta: scaled value
            logl: log(likelihood)
            blob: None

        """
        # Create a new rstate for each point, otherwise each task will generate
        # the same random number, and the rstate on master will not be incremented.
        # The argument to self.rstate.integers() is a very large integer.
        # These rstates aren't used after this map, but each time they are created,
        # a different (but deterministic) seed is used.
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
        blobs = None

        return np.array(u_list), np.array(v_list), np.array(l_list), blobs

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

    def get_nested_sampler(self, live_points, pool, pool_size):
        """
        Returns the dynested nested sampler, getting most arguments
        from the object's attributes

        Parameters
        ----------
        live_points: (numpy.ndarraym, numpy.ndarray, numpy.ndarray)
            The set of live points, in the same format as returned by
            get_initial_points_from_prior

        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation
            (pbilby implements a modified version: MPIPoolFast)

        pool_size: int
            Number of workers in the pool

        Returns
        -------
        dynesty.NestedSampler

        """
        ndim = len(self.sampling_keys)
        sampler = dynesty.NestedSampler(
            self.log_likelihood_function,
            self.prior_transform_function,
            ndim,
            pool=pool,
            queue_size=pool_size,
            periodic=self.periodic,
            reflective=self.reflective,
            live_points=live_points,
            rstate=self.rstate,
            use_pool=dict(
                update_bound=True,
                propose_point=True,
                prior_transform=True,
                loglikelihood=True,
            ),
            **self.init_sampler_kwargs,
        )

        return sampler
