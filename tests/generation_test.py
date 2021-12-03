import os
import pickle
import shutil
import unittest

from parallel_bilby import generation

GW150914_ROOT = "examples/GW150914_IMRPhenomPv2"
GW150914_INI = f"{GW150914_ROOT}/GW150914.ini"
GW150914_PSD = (
    "{H1=examples/GW150914_IMRPhenomPv2/psd_data/h1_psd.txt, "
    "L1=examples/GW150914_IMRPhenomPv2/psd_data/l1_psd.txt}"
)


def edit_ini(d):
    d = d.replace("distance-marginalization=True", "distance-marginalization=False")
    d = d.replace("{H1=psd_data/h1_psd.txt, L1=psd_data/l1_psd.txt}", GW150914_PSD)
    d = d.replace("channel_dict = {H1:GWOSC, L1:GWOSC}", "gaussian-noise = True")
    return d


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "tests/test_files/test_out"
        os.makedirs(self.outdir, exist_ok=True)
        self.ini = f"{self.outdir}/test.ini"
        ini_dat = "".join(open(GW150914_INI, "r").readlines())
        ini_dat = edit_ini(ini_dat)
        with open(self.ini, "w") as f:
            f.write(ini_dat)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_generation(self):
        generation_cli = [
            self.ini,
            "--outdir",
            self.outdir,
            "--label",
            "GW150914",
        ]
        generation.generate_runner(generation_cli)
        files = [
            "GW150914_config_complete.ini",
            "data/GW150914_data_dump.pickle",
            "submit/bash_GW150914.sh",
            "submit/analysis_GW150914_0.sh",
            "log_data_generation/GW150914.log",
        ]
        for f in files:
            path = os.path.join(self.outdir, f)
            self.assertTrue(
                os.path.isfile(path), f"After generation the file {f} not found. Files"
            )
            if "pickle" in path:
                with open(path, "rb") as file:
                    data_dump = pickle.load(file)
                    self.assertTrue(data_dump["args"].n_parallel, 4)


if __name__ == "__main__":
    unittest.main()
