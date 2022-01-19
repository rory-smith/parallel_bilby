import os.path
import shutil
import unittest

import bilby
import dill
from mpi4py import MPI
from parallel_bilby import generation


def mpi_master(func):
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
    Base test run object.
    Cannot be instantiated on its own.
    Requires the following class attributes:
        test_label
        generation_args
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

    @mpi_master
    def tearDown(self):
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
        return bilby.gw.result.CBCResult.from_json(
            os.path.join(
                self.analysis_args["outdir"],
                f"{self.analysis_args['label']}_result.json",
            )
        )
