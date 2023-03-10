import inspect
from importlib import import_module

import bilby
import bilby_pipe
import numpy as np
from bilby.core.utils import logger


def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples, sorted_samples):
    """Reorders the stored log-likelihood after they have been reweighted

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
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match."
            )
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]


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

    kwargs = dict(
        weights=None,
        roq_params=None,
        linear_matrix=None,
        quadratic_matrix=None,
        roq_scale_factor=args.roq_scale_factor,
    )
    if hasattr(args, "likelihood_roq_params") and hasattr(
        args, "likelihood_roq_weights"
    ):
        kwargs["roq_params"] = args.likelihood_roq_params
        kwargs["weights"] = args.likelihood_roq_weights
    elif hasattr(args, "roq_folder") and args.roq_folder is not None:
        logger.info(f"Loading ROQ weights from {args.roq_folder}, {args.weight_file}")
        kwargs["roq_params"] = np.genfromtxt(
            args.roq_folder + "/params.dat", names=True
        )
        kwargs["weights"] = args.weight_file
    elif hasattr(args, "roq_linear_matrix") and args.roq_linear_matrix is not None:
        logger.info(f"Loading linear_matrix from {args.roq_linear_matrix}")
        logger.info(f"Loading quadratic_matrix from {args.roq_quadratic_matrix}")
        kwargs["linear_matrix"] = args.roq_linear_matrix
        kwargs["quadratic_matrix"] = args.roq_quadratic_matrix
    return kwargs


def setup_likelihood(interferometers, waveform_generator, priors, args):
    """Takes the kwargs and sets up and returns  either an ROQ GW or GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.LALCBCWaveformGenerator
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
