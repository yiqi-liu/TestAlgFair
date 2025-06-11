"""
Utility functions.
"""
import pandas as pd
import numpy as np
import os
import git


def convert_to_log(df, col_name):
    """Convert column to log space.

    Defining log as log(x + EPSILON) to avoid division by zero.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    col_name : str
        Name of column in df to convert to log.

    Returns
    -------
    np.ndarray
        Values of column in log space

    """
    # This is to avoid division by zero while doing np.log10
    EPSILON = 1
    return np.log10(df[col_name].values + EPSILON)


def convert_to_percentile(df, col_name):
    """Convert column to percentile.

    Parameters
    ----------
    df : pd.DataFrame
        Data dataframe.
    col_name : str
        Name of column in df to convert to percentile.

    Returns
    -------
    pd.Series
        Column converted to percentile from 1 to 100

    """
    return pd.qcut(df[col_name].rank(method='first'), 100,
                   labels=range(1, 101))


def get_git_dir():
    """Get directory where git repo is saved.

    Returns
    -------
    str
        Full path of git repo home.

    """
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir


def create_dir(*args):
    """Create directory if it does not exist.

    Parameters
    ----------
    *args : type
        Description of parameter `*args`.

    Returns
    -------
    str
        Full path of directory.

    """
    fullpath = os.path.join(*args)

    # if path does not exist, create it
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    return fullpath
