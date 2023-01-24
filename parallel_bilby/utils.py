import functools
import logging
import os
import sys
import time

import bilby
from dynesty.utils import get_print_fn_args

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


def stopwatch(_func=None, *, log_level=logging.DEBUG):
    def decorator_stopwatch(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dur = time.time() - t0
            logger.log(log_level, f"{func.__name__} took {dur}s")
            return result

        return wrapper

    if _func is None:
        return decorator_stopwatch
    else:
        return decorator_stopwatch(_func)


def stdout_sampling_log(**kwargs):
    """Logs will look like:
    #:282|eff(%):26.406|logl*:-inf<-160.2<inf|logz:-165.5+/-0.1|dlogz:1038.1>0.1

    Adapted from dynesty
    https://github.com/joshspeagle/dynesty/blob/bb1c5d5f9504c9c3bbeffeeba28ce28806b42273/py/dynesty/utils.py#L349
    """
    niter, short_str, mid_str, long_str = get_print_fn_args(**kwargs)
    custom_str = [f"#: {niter:d}"] + mid_str
    custom_str = "|".join(custom_str).replace(" ", "")
    sys.stdout.write("\033[K" + custom_str + "\r")
    sys.stdout.flush()
