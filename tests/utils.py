import os.path
import shutil
import signal
import unittest

import bilby
import dill
from mpi4py import MPI
from parallel_bilby import generation


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
    def analysis_args(self):
        return dict(
            data_dump=os.path.join(
                self.test_dir, "data", f"{self.test_label}_data_dump.pickle"
            ),
            outdir=os.path.join(self.test_dir, "result"),
            label=self.test_label,
        )

    @mpi_master
    def setUp(self):
        generation.generate_runner(
            outdir=self.test_dir,
            label=self.test_label,
            **self.generation_args,
        )

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
        with open(
            os.path.join(
                self.analysis_args["outdir"],
                f"{self.analysis_args['label']}_checkpoint_resume.pickle",
            ),
            "rb",
        ) as f:
            resume_file = dill.load(f)
        return resume_file

    def read_bilby_result(self):
        return bilby.gw.result.CBCResult.from_hdf5(
            os.path.join(
                self.analysis_args["outdir"],
                f"{self.analysis_args['label']}_result.hdf5",
            )
        )


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
