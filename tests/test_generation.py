import os.path
import pickle

import pytest
from tests.cases import GW150914Run


class GenerationTest(GW150914Run):
    @pytest.mark.mpi_skip
    def test_generation(self):
        files = [
            "GW150914_config_complete.ini",
            "data/GW150914_data_dump.pickle",
            "log_data_generation/GW150914.log",
        ]
        for f in files:
            path = os.path.join(self.test_dir, f)
            self.assertTrue(
                os.path.isfile(path), f"After generation the file {f} not found. Files"
            )
            if "pickle" in path:
                with open(path, "rb") as file:
                    data_dump = pickle.load(file)
                    self.assertTrue(data_dump["args"].n_parallel, 4)
