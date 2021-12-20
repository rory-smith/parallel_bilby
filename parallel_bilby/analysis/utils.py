import numpy as np


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
