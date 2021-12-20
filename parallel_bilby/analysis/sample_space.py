import numpy as np
from bilby.gw import conversion


def fill_sample(args):
    ii, sample, likelihood = args
    sample = dict(sample).copy()
    marg_params = likelihood.parameters.copy()
    likelihood.parameters.update(sample)
    sample.update(likelihood.get_sky_frame_parameters())
    sample = likelihood.generate_posterior_sample_from_marginalized_likelihood()
    # Likelihood needs to have marg params to calculate correct SNR
    likelihood.parameters.update(marg_params)
    conversion.compute_snrs(sample, likelihood)
    sample = conversion._generate_all_cbc_parameters(
        sample,
        likelihood.waveform_generator.waveform_arguments,
        conversion.convert_to_lal_binary_black_hole_parameters,
    )
    return sample


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


def get_initial_points_from_prior(
    ndim,
    npoints,
    prior_transform_function,
    log_prior_function,
    log_likelihood_function,
    pool,
    rstate,
    calculate_likelihood=True,
):

    # Create a new rstate for each point, otherwise each task will generate
    # the same random number, and the rstate on master will not be incremented
    sg = np.random.SeedSequence(rstate.integers(9223372036854775807))
    map_rstates = [np.random.Generator(np.random.PCG64(n)) for n in sg.spawn(npoints)]

    args_list = [
        (
            prior_transform_function,
            log_prior_function,
            log_likelihood_function,
            ndim,
            calculate_likelihood,
            map_rstates[i],
        )
        for i in range(npoints)
    ]
    initial_points = pool.map(get_initial_point_from_prior, args_list)
    u_list = [point[0] for point in initial_points]
    v_list = [point[1] for point in initial_points]
    l_list = [point[2] for point in initial_points]

    return np.array(u_list), np.array(v_list), np.array(l_list)
