import os
import sys


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
