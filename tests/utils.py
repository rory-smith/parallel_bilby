import logging
import os.path
import shutil
import signal
import unittest

import bilby
import dill
from mpi4py import MPI
from parallel_bilby import generation

logger = logging.getLogger("pBilbyTesting")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def mpi_master(func):
    """
    Custom decorator for running functions/methods on the master MPI task only.
    """

    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            f = func(*args, **kwargs)
        else:
            f = None
        comm.Barrier()
        return f

    return wrapper


class _Run(unittest.TestCase):
    """
    Base test run object. Runs the generation during setup,
    and tears down the directory tree created by generation
    at the end of testing.

    Also provides methods for reading the resume file and
    reading the results.

    Note: it CANNOT be instantiated on its own!
    Requires the following class attributes:
        test_label: str
            A simple label for the test (with no spaces).
            This will also be used to create the test directory.
        generation_args: dict
            Dictionary of options to be used for generation.
            These options can be anything you would normally
            specify via the CLI or in the generation ini file.

    """

    @property
    def test_dir(self):
        return os.path.join("/tmp/pbilby_test", self.test_label)

    @property
    def outdir(self):
        return self.test_dir

    @property
    def result_dir(self):
        return os.path.join(self.test_dir, "result")

    @property
    def data_dump_path(self):
        return os.path.join(
            self.test_dir, "data", f"{self.test_label}_data_dump.pickle"
        )

    @property
    def analysis_args(self):
        kwargs = dict(
            data_dump=self.data_dump_path,
            outdir=self.result_dir,
            label=self.test_label,
            dynesty_sample="acceptance-walk",
            nlive=5,
            dynesty_bound="live",
            walks=100,
            maxmcmc=5000,
            naccept=60,
            nact=2,
            facc=0.5,
            min_eff=10,
            enlarge=1.5,
            sampling_seed=0,
            proposals=None,
            bilby_zero_likelihood_mode=False,
        )
        for key in kwargs.keys():
            if key in self.generation_args:
                kwargs[key] = self.generation_args[key]
        kwargs["outdir"] = self.result_dir
        logger.debug(f"ANALYSIS-RUN kwargs: {kwargs}")
        return kwargs

    @property
    def analysis_runner_kwargs(self):
        kwargs = dict(
            data_dump=self.data_dump_path,
            outdir=self.result_dir,
            label=self.test_label,
            **self.generation_args,
        )
        logger.debug(f"ANALYSIS-RUNNER kwargs: {kwargs}")
        return kwargs

    @mpi_master
    def setUp(self):
        kwargs = dict(
            outdir=self.test_dir,
            label=self.test_label,
            **self.generation_args,
        )
        logger.debug(f"Running generation with kwargs: {kwargs}")
        generation.generate_runner(**kwargs)

    def tearDown(self):
        # Make sure all MPI tasks are here
        # If any are stuck, fail the test and kill the test
        comm = MPI.COMM_WORLD
        with timeout(seconds=10):
            comm.Barrier()

        # Delete test directory
        if comm.Get_rank() == 0:
            shutil.rmtree(self.test_dir)

    def read_resume_file(self):
        resume_file = os.path.join(
            self.result_dir, f"{self.test_label}_checkpoint_resume.pickle"
        )
        if not os.path.isfile(resume_file):
            logger.error(
                f"Resume file {resume_file} does not exist:\n{self.outdir_filetree}"
            )
        with open(resume_file, "rb") as f:
            resume_file = dill.load(f)
        return resume_file

    @property
    def outdir_filetree(self):
        return dirtree(self.outdir)

    def read_bilby_result(self):
        result_path = os.path.join(self.result_dir, f"{self.test_label}_result.hdf5")
        if not os.path.isfile(result_path):
            logger.error(
                f"Result file {result_path} does not exist:\n{self.outdir_filetree}"
            )
        return bilby.gw.result.CBCResult.from_hdf5(result_path)


class timeout:
    """
    Custom context for handling MPI tasks that might hang during testing.
    """

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        print(
            f"Calling MPI_Abort to end test early because a task is stuck (timeout={self.seconds})"
        )
        MPI.COMM_WORLD.Abort()

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def dirtree(root):
    str = ""
    for root, dirs, files in os.walk(root):
        for d in dirs:
            str += f"{os.path.join(root, d)}\n"
        for f in files:
            str += f"{os.path.join(root, f)}\n"
    return str
