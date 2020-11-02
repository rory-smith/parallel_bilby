"""
Module to create an injection file + pbilby inis for the injections.
"""
import logging
import os
import shutil

from bilby_pipe.create_injections import create_injection_file

logging.getLogger().setLevel(logging.INFO)


N_INJECTION = 100
LABEL = "bbh"  # the main name of the injections
INJECTION_FILE = f"./datafiles/{LABEL}_injections.json"
PRIOR_FILE = "./datafiles/bbh.prior"


def create_ini(injection_idx: int):
    unique_label = f"{LABEL}_{injection_idx}"
    outdir = f"out_{unique_label}"
    ini = f"{unique_label}.ini"
    with open("pbilby_config_template.ini", "r") as f:
        txt = f.read()
        txt = txt.replace("{{{IDX}}}", str(injection_idx))
        txt = txt.replace("{{{LABEL}}}", unique_label)
        txt = txt.replace("{{{OUTDIR}}}", outdir)
        txt = txt.replace("{{{PRIOR_FILE}}}", PRIOR_FILE)
        txt = txt.replace("{{{INJECTION_FILE}}}", INJECTION_FILE)
    with open(ini, "w") as f:
        f.write(txt)


def create_data_generation_slurm_submission_file():
    os.makedirs("generation_log", exist_ok=True)
    with open("pbilby_generation_job_template.ini", "r") as f:
        txt = f.read()
        txt = txt.replace("{{{GENERATION_LOG_DIR}}}", "generation_log")
        txt = txt.replace("{{{NUM_INJECTIONS}}}", str(N_INJECTION))
        txt = txt.replace("{{{LABEL}}}", LABEL)
        txt = txt.replace(
            "{{{GENERATION_EXE}}}", shutil.which("parallel_bilby_generation")
        )
    with open("slurm_data_generation.sh", "w") as f:
        f.write(txt)


def create_analysis_bash_runner():
    file_contents = "#! /bin/sh\n"
    for i in range(N_INJECTION):
        label = f"{LABEL}_{i}"
        analysis_file = f"out_{label}/submit/bash_{label}.sh"
        file_contents += f"bash {analysis_file}\n"
    with open("start_data_analysis.sh", "w") as f:
        f.write(file_contents)


def main():
    logging.info("Generating injection file")
    create_injection_file(
        filename=INJECTION_FILE,
        prior_file=PRIOR_FILE,
        n_injection=N_INJECTION,
        generation_seed=0,
    )

    logging.info("Generating parallel bilby ini files + submission scripts")
    for i in range(N_INJECTION):
        create_ini(injection_idx=i)

    create_data_generation_slurm_submission_file()
    logging.info("Start generation jobs with:\n$ sbatch slurm_data_generation.sh")

    create_analysis_bash_runner()
    logging.info(
        "After generation, begin analysis with:\n$ bash start_data_analysis.sh"
    )


if __name__ == "__main__":
    main()
