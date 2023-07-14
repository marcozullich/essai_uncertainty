import torch
from torch import nn

class SingleNetworkPointwiseBNN(nn.Module):
  '''
  A generic class for implementic Bayesian Neural Networks composed one point-
  wise NN with a stochastic component (e.g., DropConnect or Dropout).

  Params
  --------
  module: the underlying neural network, which should contain at least one
    Dropout layer
  n_eval: number of evaluations to be produced when calling forward in eval mode

  '''
  def __init__(self, module, n_eval):
    super().__init__()
    self.module = module
    self.n_eval = n_eval

  def forward(self, data):
    if self.training:
      return self.module(data)
    new_shape = [self.n_eval, *[1]*len(data.shape)]
    data_rep = data.repeat(new_shape)
    return self.module(data_rep).squeeze()

class BaseVariationalLayer(nn.Module):
  def __init__(self, parent_module):
    super().__init__()
    self.parent_module = parent_module
    if not hasattr(self.parent_module, "kl_accumulator"):
      self.parent_module.kl_accumulator = 0.0

  def accumulate_kl_divergence(self, value):
    self.parent_module.kl_accumulator += value

  def gaussian_kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):
    '''
    Computes KL divergence between two Gaussian distributions: KL(Q||P)
    '''
    kl = torch.log(sigma_p) - torch.log(sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
    return kl.mean()

class VariationalModel(nn.Module):
  def __init__(self, module, n_eval, n_batches_per_epoch):
    super().__init__()
    self.module = module
    self.n_eval = n_eval
    self.set_regularizing_factor_children(n_batches_per_epoch)
    self.reset_kl_accumulator()


  def set_regularizing_factor_children(self, n_batches):
    for module in self.module.modules():
      if hasattr(module, "set_n_batches_per_epoch"):
        module.set_n_batches_per_epoch(n_batches)

  def reset_kl_accumulator(self):
    self.kl_accumulator = 0

  def forward(self, data):
    return self.module(data)


