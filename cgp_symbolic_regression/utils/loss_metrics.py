# !/usr/bin/env python

"""

"""
import numpy as np
import math
from sklearn.metrics import matthews_corrcoef, mean_squared_error

from cgp_symbolic_regression.utils.params import params


def mcc(model_output, target):
    tp = np.sum(np.logical_and(model_output == params["true_val"], target == params["true_val"]), dtype=np.float32)
    tn = np.sum(np.logical_and(model_output == params["neg_val"], target == params["neg_val"]), dtype=np.float32)
    fp = np.sum(np.logical_and(model_output == params["true_val"], target == params["neg_val"]), dtype=np.float32)
    fn = np.sum(np.logical_and(model_output == params["neg_val"], target == params["true_val"]), dtype=np.float32)
    if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) <= 1e-5:
        return 1

    mcc_value = (tp * tn - fp * fn) / math.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 1 - abs(mcc_value)


def summed_loss(model_output, target):
    return mean_squared_error(y_true=target, y_pred=model_output)


def mcc_multiclass(model_output, target):
    return 1 - abs(matthews_corrcoef(y_true=target, y_pred=model_output))
