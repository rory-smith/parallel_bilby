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
    "-n", "--nsamples", type=int, default=5000,
    help="If enough samples available, resample to this number of samples")
args = parser.parse_args()

result = bilby.gw.result.CBCResult.from_json(args.result)

with open(result.meta_data["data_dump"], "rb") as file:
    data_dump = pickle.load(file)
    likelihood = data_dump["likelihood"]

priors = result.priors
posterior = result.posterior

if args.nsamples is not False and len(posterior) > args.nsamples:
    posterior = posterior.sample(args.nsamples)
else:
    pass

result.label = result.label + "_post3"

for key, val in priors.items():
    cond1 = key not in posterior
    cond2 = val.is_fixed
    cond3 = not isinstance(val, bilby.core.prior.Constraint)
    if cond1 and cond2 and cond3:
        posterior[key] = val.peak

posterior, _ = conversion.convert_to_lal_binary_neutron_star_parameters(
    posterior)
posterior = conversion.generate_mass_parameters(posterior)
posterior = conversion.generate_tidal_parameters(posterior)


new_time_samples = list()
new_distance_samples = list()
new_phase_samples = list()
t0 = datetime.datetime.now()
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    print(f"Starting to fill posterior with POOL={pool.size}")
    samples = np.array(pool.map(generate_sample, range(len(posterior))))
    posterior['geocent_time'] = samples[:, 0]
    posterior['luminosity_distance'] = samples[:, 1]
    posterior['phase'] = samples[:, 2]
    dt = datetime.datetime.now() - t0
    print(f"Finished, time taken = {dt}")

    result.posterior = posterior
    result.save_to_file(extension="json")
