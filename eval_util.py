from pathlib import Path
from warnings import warn
import hashlib
from collections import namedtuple
import sklearn.preprocessing as pre
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sed_eval
from sklearn.metrics import auc
from psds_eval import PSDSEval, plot_psd_roc
from psds_eval.psds import WORLD, PSDSEvalError
#from sed_scores_eval import intersection_based


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def binarize(x, threshold=0.5):
    if x.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in x])
    else:
        return pre.binarize(x, threshold=threshold)


def median_filter(x, window_size, threshold=0.5):
    x = binarize(x, threshold=threshold)
    if x.ndim == 3: # (batch_size, time_steps, num_classes)
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1: # (batch_size, time_steps)
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1: # (time_steps, num_classes)
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def predictions_to_time(df, ratio):
    if len(df) == 0:
        return df
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df

