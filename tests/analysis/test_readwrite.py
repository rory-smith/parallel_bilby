import multiprocessing
import os
import shutil

import dill
from deepdiff import DeepDiff
from parallel_bilby import generation
from parallel_bilby.analysis import analysis_run, read_write

outdir = "tests/test_files/out_readwrite_test/"
reference_file = "tests/test_files/test_readwrite_sampler_reference.dill"


def create_test_data():
    """Only needs to be run once to generate reference data.
    This is not run during the test - it has to be done manually."""

    # Use same ini file as fast e2e test
    generation.generate_runner(["tests/test_files/fast_test.ini", "--outdir", outdir])

    run = analysis_run.AnalysisRun(
        data_dump=os.path.join(outdir, "data/fast_injection_data_dump.pickle"),
        outdir=outdir,
    )

    # Create a test pool
    pool = multiprocessing.Pool(4)

    # Generate live points
    live_points = run.get_initial_points_from_prior(pool)

    # Get sampler object and write reference pickle
    sampler = run.get_nested_sampler(live_points, pool=pool, pool_size=1)

    with open(reference_file, "wb") as dill_file:
        dill.dump(sampler, dill_file)

    # Clean up
    shutil.rmtree(outdir)


def test_readwrite():
    # Read reference data
    with open(reference_file, "rb") as dill_file:
        sampler_ref = dill.load(dill_file)

    # Dynesty strips these attributes but does not catch errors properly
    # if they are already missing, so we have to set them to None
    sampler_ref.rstate = None
    sampler_ref.pool = None
    sampler_ref.loglikelihood.pool = None

    filename = "readwrite_test_file"
    sampling_time_ref = 123.0

    try:
        # Write to file
        read_write.write_current_state(
            sampler_ref, filename, sampling_time_ref, rotate=False
        )

        # Read it again
        sampler, sampling_time = read_write.read_saved_state(filename)

    finally:
        # Delete file
        os.remove(filename)

    assert sampling_time == sampling_time_ref

    assert {} == DeepDiff(
        sampler.__getstate__(),
        sampler_ref.__getstate__(),
        exclude_paths=[
            "root['loglikelihood'].pool",
            "root['kwargs']['sampling_time']",
            "root['nqueue']",
        ],
    )
