#!/usr/bin/env python
"""
"""
import os
import sys
import argparse
import pickle
from schwimmbad import MPIPool
import datetime

import numpy as np
import bilby
from bilby.gw import conversion

logger = bilby.core.utils.logger

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "0"
os.environ["MPI_PER_NODE"] = "16"


def main():
    """ Do nothing function to play nicely with MPI """
    pass


def generate_sample(ii):
    sample = dict(posterior.iloc[ii]).copy()
    likelihood.parameters.update(sample)
    samp = likelihood.generate_posterior_sample_from_marginalized_likelihood()
    new_time_sample = samp['geocent_time']
    new_distance_sample = samp['luminosity_distance']
    new_phase_sample = samp['phase']
    return new_time_sample, new_distance_sample, new_phase_sample


parser = argparse.ArgumentParser()
parser.add_argument("result", type=str)
parser.add_argument(
    "-n", "--nsamples", type=int, default=False,
    help="If enough samples available, resample to this number of samples")
parser.add_argument(
    "--rand-seed", type=int, default=1234,
    help="Random seed: important for reproducible resampling")

args = parser.parse_args()

np.random.seed(args.rand_seed)

result = bilby.gw.result.CBCResult.from_json(args.result)

with open(result.meta_data["data_dump"], "rb") as file:
    data_dump = pickle.load(file)
    likelihood = data_dump["likelihood"]

priors = result.priors
posterior = result.posterior

if args.nsamples is not False and len(posterior) > args.nsamples:
    logger.debug("Downsampling to {} samples".format(args.nsamples))
    posterior = posterior.sample(args.nsamples)
else:
    pass

for key, val in priors.items():
    cond1 = key not in posterior
    cond2 = val.is_fixed
    cond3 = not isinstance(val, bilby.core.prior.Constraint)
    if cond1 and cond2 and cond3:
        posterior[key] = val.peak

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    result.label = result.label.relabel("_pre", "_postprocessed")

    if data_dump["args"].binary_neutron_star:
        logger.info("Using BNS source model")
        posterior, _ = conversion.convert_to_lal_binary_neutron_star_parameters(
            posterior)
        posterior = conversion.generate_all_bns_parameters(posterior)
    else:
        logger.info("Using BBH source model")
        posterior, _ = conversion.convert_to_lal_binary_black_hole_parameters(
            posterior)
        posterior = conversion.generate_all_bbh_parameters(posterior)

    logger.info("Updating prior to the actual prior")
    for par, name in zip(
            ['distance', 'phase', 'time'],
            ['luminosity_distance', 'phase', 'geocent_time']):
        if getattr(likelihood, '{}_marginalization'.format(par), False):
            priors[name] = likelihood.priors[name]
    result.priors = priors

    new_time_samples = list()
    new_distance_samples = list()
    new_phase_samples = list()
    t0 = datetime.datetime.now()

    logger.info("Generating posterior from marginalized parameters")
    logger.info(f"Nsamples={len(posterior)}, POOL={pool.size}")
    samples = np.array(pool.map(generate_sample, range(len(posterior))))
    posterior['geocent_time'] = samples[:, 0]
    posterior['luminosity_distance'] = samples[:, 1]
    posterior['phase'] = samples[:, 2]
    dt = datetime.datetime.now() - t0
    logger.info(f"Finished, time taken = {dt}")

    result.posterior = posterior
    result.save_to_file(extension="json")
