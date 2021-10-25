#!/usr/bin/env python

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup

# check that python version is 3.5 or above
python_version = sys.version_info
print("Running Python version %s.%s.%s" % python_version[:3])
if python_version < (3, 5):
    sys.exit("Python < 3.5 is not supported, aborting setup")
print("Confirmed Python version 3.5.0 or above")


def write_version_file(version):
    """Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file (relative to the src package directory)
    """
    version_file = Path("parallel_bilby") / ".version"

    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:  # git calls failed
        # we already have a version file, let's use it
        if version_file.is_file():
            return version_file.name
        # otherwise error out
        exc.args = (
            f"unable to obtain git version information, and {version_file} doesn't "
            f"exist, cannot continue ({exc})",
        )
        raise
    else:
        clean_status = ("UNCLEAN" if git_diff else "CLEAN",)
        git_version = f"{version}: ({clean_status}) {git_log.rstrip()}"
        print(f"parsed git version info as: {git_version!r}")

    with open(version_file, "w") as f:
        print(git_version, file=f)
        print(f"created {version_file}")

    return version_file.name


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()
    return long_description


VERSION = "1.0.1"
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(
    name="parallel_bilby",
    description="Running bilby at scale",
    long_description=long_description,
    url="https://git.ligo.org/gregory.ashton/parallel_bilby",
    author="Gregory Ashton, Rory Smith, Avi Vajpeyi",
    author_email="gregory.ashton@ligo.org",
    license="MIT",
    version=VERSION,
    packages=["parallel_bilby"],
    package_dir={"parallel_bilby": "parallel_bilby"},
    package_data={"parallel_bilby": [version_file]},
    install_requires=[
        "future",
        "bilby>=1.1.1",
        "bilby_pipe>=1.0.3",
        "scipy>=1.2.0",
        "gwpy",
        "matplotlib",
        "numpy",
        "tqdm",
        "corner",
        "dynesty>=1.0.0",
        "schwimmbad",
        "pandas",
        "nestcheck",
        "mpi4py>3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "parallel_bilby_generation=parallel_bilby.generation:main",
            "parallel_bilby_analysis=parallel_bilby.analysis:main",
            "parallel_bilby_ptemcee_analysis=parallel_bilby.ptemcee_analysis:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
)
