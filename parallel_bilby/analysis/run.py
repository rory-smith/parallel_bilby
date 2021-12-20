import inspect
import logging
import os
import pickle
from importlib import import_module

import bilby
import bilby_pipe
import dynesty
import numpy as np
from bilby.core.utils import logger


def roq_likelihood_kwargs(args):
    """Return the kwargs required for the ROQ setup

    Parameters
    ----------
    args: Namespace
        The parser arguments

    Returns
    -------
    kwargs: dict
        A dictionary of the required kwargs

    """
    if hasattr(args, "likelihood_roq_params"):
        params = args.likelihood_roq_params
    else:
        params = np.genfromtxt(args.roq_folder + "/params.dat", names=True)

    if hasattr(args, "likelihood_roq_weights"):
        weights = args.likelihood_roq_weights
    else:
        weights = args.weight_file
        logger.info(f"Loading ROQ weights from {weights}")

    return dict(
        weights=weights, roq_params=params, roq_scale_factor=args.roq_scale_factor
    )


def setup_likelihood(interferometers, waveform_generator, priors, args):
    """Takes the kwargs and sets up and returns  either an ROQ GW or GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        The waveform generation
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments


    Returns
    -------
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        The likelihood (either GravitationalWaveTransient or ROQGravitationalWaveTransient)

    """

    likelihood_kwargs = dict(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=args.phase_marginalization,
        distance_marginalization=args.distance_marginalization,
        distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
        time_marginalization=args.time_marginalization,
        reference_frame=args.reference_frame,
        time_reference=args.time_reference,
    )

    if args.likelihood_type == "GravitationalWaveTransient":
        Likelihood = bilby.gw.likelihood.GravitationalWaveTransient
        likelihood_kwargs.update(jitter_time=args.jitter_time)

    elif args.likelihood_type == "ROQGravitationalWaveTransient":
        Likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient

        if args.time_marginalization:
            logger.warning(
                "Time marginalization not implemented for "
                "ROQGravitationalWaveTransient: option ignored"
            )

        likelihood_kwargs.pop("time_marginalization", None)
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args))
    elif "." in args.likelihood_type:
        split_path = args.likelihood_type.split(".")
        module = ".".join(split_path[:-1])
        likelihood_class = split_path[-1]
        Likelihood = getattr(import_module(module), likelihood_class)
        likelihood_kwargs.update(
            bilby_pipe.utils.convert_string_to_dict(args.extra_likelihood_kwargs)
        )
        if "roq" in args.likelihood_type.lower():
            likelihood_kwargs.pop("time_marginalization", None)
            likelihood_kwargs.pop("jitter_time", None)
            likelihood_kwargs.update(args.roq_likelihood_kwargs)
    else:
        raise ValueError("Unknown Likelihood class {}")

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    logger.info(
        f"Initialise likelihood {Likelihood} with kwargs: \n{likelihood_kwargs}"
    )

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood


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
