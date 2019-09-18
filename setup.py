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
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file (relative to the src package directory)
    """
    version_file = Path("src") / ".version"

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
            "unable to obtain git version information, and {} doesn't "
            "exist, cannot continue ({})".format(version_file, str(exc)),
        )
        raise
    else:
        git_version = "{}: ({}) {}".format(
            version, "UNCLEAN" if git_diff else "CLEAN", git_log.rstrip()
        )
        print("parsed git version info as: {!r}".format(git_version))

    with open(version_file, "w") as f:
        print(git_version, file=f)
        print("created {}".format(version_file))

    return version_file.name


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()
    return long_description


VERSION = "0.0.1"
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(
    name="parallel_bilby",
    description="Running bilby at scale",
    long_description=long_description,
    url="https://git.ligo.org/gregory.ashton/parallel_bilby",
    author="Gregory Ashton, Rory Smith",
    author_email="gregory.ashton@ligo.org",
    license="MIT",
    version=VERSION,
    packages=["parallel_bilby"],
    package_dir={'parallel_bilby': 'src'},
    install_requires=[
        "future",
        "bilby>=0.5.4",
        "scipy>=1.2.0",
        "gwpy",
        "matplotlib",
        "numpy",
        "tqdm",
        "corner",
        "dynesty>=0.9.7",
        "schwimmbad",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "parallel_bilby_generation=parallel_bilby.generation:main",
            "parallel_bilby_analysis=parallel_bilby.analysis:main",
            "parallel_bilby_postprocess=parallel_bilby.postprocess:main",
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
