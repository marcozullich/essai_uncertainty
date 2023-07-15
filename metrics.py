import numpy as np

def accuracy_ensemble(prediction, ground_truth, axis_ensemble=0):
    ensemble_prediction = np.mean(prediction, axis=axis_ensemble)
    prediction_classes = ensemble_prediction.argmax(1)
    return np.mean(np.equal(prediction_classes, ground_truth))

