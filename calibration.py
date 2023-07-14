import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

def accuracy_prob_models(predictions, ground_truth, preds_as_logits=False, axis_classes=2):
  if preds_as_logits:
    predictions = softmax(predictions, axis_classes)
  predictions_mean_per_sample = predictions.mean(axis=1)
  assignment_class = predictions_mean_per_sample.argmax(axis=1)
  return (assignment_class == ground_truth).mean()

def confidence_prob_models(predictions, preds_as_logits=False, axis_classes=2):
  if preds_as_logits:
    predictions = softmax(predictions, axis_classes)
  predictions_mean_per_sample = predictions.mean(axis=1)
  confidence_per_datapoint = predictions_mean_per_sample.max(axis=1)
  return confidence_per_datapoint

def confidence_binning(confidence_vector, n_bins=10):
  bins = np.linspace(1/n_bins, 1, n_bins)
  return np.digitize(confidence_vector, bins), bins

def reliability_vector(predictions, ground_truth, n_bins=10, preds_as_logits=False, axis_classes=2):
  if preds_as_logits:
    predictions = softmax(predictions, axis_classes)
  confidence_scores = confidence_prob_models(predictions)

  bins_composition, bins_cutoffs = confidence_binning(confidence_scores, n_bins)

  mean_accuracy_per_bins = np.full((n_bins,), fill_value=np.nan)
  bin_counts = np.bincount(bins_composition)

  for i in range(n_bins):
    if i > bins_composition.max():
      break
    if bin_counts[i] > 0:
      group_accuracy = accuracy_prob_models(
          predictions[bins_composition==i],
          ground_truth[bins_composition==i]
        )
      mean_accuracy_per_bins[i] = group_accuracy

  return mean_accuracy_per_bins, bins_cutoffs

def reliability_plot(reliability_vector, bins_cutoffs, clear_nans=True):
  bins_delta = bins_cutoffs[1] - bins_cutoffs[0]
  x_axis = bins_cutoffs - bins_delta/2

  if clear_nans:
    x_axis = x_axis[~np.isnan(reliability_vector)]
    reliability_vector = reliability_vector[~np.isnan(reliability_vector)]

  fig, ax = plt.subplots()
  ax.scatter(
      x_axis,
      reliability_vector
  )
  ax.set_xlim((0,1))
  ax.set_ylim((0,1))
  line = mlines.Line2D([0, 1], [0, 1], color='red')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  plt.plot(x_axis, reliability_vector)
  plt.show()