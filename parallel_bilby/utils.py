import datetime
import os
import sys
import timeit

import bilby

logger = bilby.core.utils.logger


def get_cli_args():
    """Tool to get CLI args (also makes testing easier)"""
    return sys.argv[1:]


def get_version_information():
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "parallel_bilby/.version"
    )
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


def safe_file_dump(data, filename, module):
    """Safely dump data to a .pickle file

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    os.rename(temp_filename, filename)


def stopwatch(method):
    """A decorator that logs the time spent in a function"""

    def timed(*args, **kw):
        t_start = timeit.time.perf_counter()
        result = method(*args, **kw)
        t_end = timeit.time.perf_counter()
        duration = datetime.timedelta(seconds=t_end - t_start)
        logger.info(f"{method.__name__}: {duration}")
        return result

    return timed
