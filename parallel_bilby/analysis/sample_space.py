from bilby.gw import conversion


def fill_sample(args):
    """Fill the sample for a particular row in the posterior data frame.

    This function is used inside a pool.map(), so its interface needs
    to be a single argument that is then manually unpacked.

    Parameters
    ----------
    args: tuple
        (row number, row, likelihood)

    Returns
    -------
    sample: array-like

    """
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
