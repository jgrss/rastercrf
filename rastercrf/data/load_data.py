import os
from pathlib import Path

import joblib


def load_data(name):

    """
    Loads data from file

    Args:
        name (str): The data name to load.

    Returns:
        ``tuple``
    """

    file_path = Path(os.path.dirname(os.path.realpath(__file__)))

    file_name = str(file_path.joinpath(name + '.bz2'))

    return joblib.load(file_name)
