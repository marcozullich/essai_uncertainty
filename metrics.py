import numpy as np
import sklearn.metrics as M

def accuracy_ensemble(prediction, ground_truth, axis_ensemble=0, **kwargs):
    return M.accuracy_score(ground_truth, prediction.mean(axis=axis_ensemble), **kwargs)

