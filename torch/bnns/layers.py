import torch
from torch import nn
from .base import BaseVariationalLayer

class MCDropout(nn.Dropout):
  '''
  A simple extension of Dropout which just overrides the train() call.
  In this way, simply calling .train(False) on Modules making use of MCDropout
  or .eval() will not trigger the eval behaviour of Dropout
  '''
  def train(self, train=True):
    self.training = True

class LinearMCDropConnect(nn.Linear):
  def __init__(self, drop_connect_p, **kwargs):
    super().__init__(**kwargs)
    self.p = drop_connect_p
    self.dropout = nn.Dropout(self.p)

  def train(self, training=True):
    self.training = True

  def forward(self, data):
    w = self.dropout(self.weight)
    b = self.dropout(self.bias)
    return torch.nn.functional.linear(data, w, b)

class LinearVariational(BaseVariationalLayer):
  '''
  A redesigning of the Linear Module for variational inference with Gaussian
  variational posterior, which are parameterized by mu (mean) and rho.
  rho is linked to the std by the softplus operator, which is applied to rho
  to ensure that the std of the Gaussian is always positive, which
  The priors for mu and rho are both Gaussian with tunable mean and std.


  Parameters
  ------------
  in_features: the number of input features (neurons) from the previous layer.
  out_features: the number of output features (neurons) for this layer.
  parent_module: the father module of the current one, needed for reasons of
    KL divergence calculation
  #### batches_per_epoch: the number of batches per epoch for rescaling purposes.
  bias: a boolean indicating whether the current layer has bias terms (def True).
  prior_mean_mu: the mean of the prior Gaussian for mu
  '''
  def __init__(self, in_features, out_features, parent_module,
               bias=True, prior_mean_mu=0.0, prior_std_mu=0.01,
               prior_mean_rho=-3.0, prior_std_rho=0.01):
    super().__init__(parent_module)

    # the parameters for mean and std of the weights follow a standard Gaussian
    self.mu_weights = torch.normal(prior_mean_mu, prior_std_mu, (out_features, in_features))
    self.rho_weights = torch.normal(prior_mean_rho, prior_std_rho, (out_features,in_features))

    # the parameters for mean and std of the
    self.mu_bias = torch.zeros((out_features,))
    self.rho_bias = torch.zeros((out_features,))

  def reparameterize(self, mu, rho):
    sigma = torch.log(1 + torch.exp(rho))
    eps = torch.randn_like(sigma)
    return mu + (eps * sigma)

  # def compute_KL_divergence(self, z, mu, sigma, regularizing_factor=1.0):
  #   log_posterior = torch.dist.Normal(mu, sigma).log_prob(z)
  #   log_prior = torch.dist.Normal(0, 1).log_prob(z)
  #   return (log_posterior - log_prior).sum() / regularizing_factor

  def set_n_batches_per_epoch(self, n_batches_per_epoch):
    self.kl_regularizing_factor = 1/n_batches_per_epoch

  def forward(self, data):
    sigma_weights = torch.nn.functional.softplus(self.rho_weights)
    weights = self.reparameterize(self.mu_weights, self.rho_weights)

    sigma_bias = torch.nn.functional.softplus(self.rho_bias)
    bias = self.reparameterize(self.mu_bias, self.rho_bias)
    z = torch.dot(data, weights) + bias

    self.parent.kl_accumulator += self.compute_KL_divergence(
        z, self.mu_weights, sigma_weights, regularizing_factor=self.kl_regularizing_factor
    )

    self.parent.kl_accumulator += self.compute_KL_divergence(
        z, self.mu_bias, sigma_bias, regularizing_factor=self.kl_regularizing_factor
    )

    return z

class LinearFlipout(BaseVariationalLayer):
  def __init__(
      self,
      in_features,
      out_features,
      parent_module,
      prior_mean=0,
      prior_variance=1,
      posterior_mu_init=0,
      posterior_rho_init=-3.0,
      std_init=0.1
  ):
    super().__init__(self, parent_module)
    self.in_features = in_features
    self.out_features = out_features
    self.prior_mean = prior_mean
    self.prior_variance = prior_variance
    self.posterior_mu_init = posterior_rho_init

    # trainable parameters
    self.mu = torch.nn.Parameter(torch.Tensor((out_features, in_features)))
    self.rho = torch.nn.Parameter(torch.Tensor((out_features, in_features)))

    # non-trainable parameters
    self.register_buffer(
        "epsilon", torch.Tensor((out_features, in_features)), persistent=False
    )
    self.register_buffer(
        "prior_mu", torch.Tensor((out_features, in_features)), persistent=False
    )
    self.register_buffer(
        "prior_sigma", torch.Tensor((out_features, in_features)), persistent=False
    )

    # parameters initialization
    # priors -> fill with constant values
    self.prior_mu.fill_(self.prior_mean)
    self.prior_sigma.fill_(self.prior_variance)

    # init weight and base perturbation weights
    self.mu.data.normal_(mean=self.posterior_mu_init, std=std_init)
    self.rho.data.normal_(mean=self.posterior_rho_init, std=std_init)

  def forward(self, data):
    # compute sigma from rho
    sigma = torch.nn.functional.softplus(self.rho)

    # in-place sampling of epsilon
    epsilon = self.epsilon.data.normal_()
    delta = sigma * epsilon

    self.accumulate_kl_divergence(self.gaussian_kl_divergence(
        self.mu, sigma, self.prior_mu, self.prior_sigma
    ))

    # unperturbed outputs
    y_pred_unperturbed = torch.nn.functional.linear(data, self.mu)

    # create perturbations
    sign_input = torch.empty_like(data).uniform_(-1, 1).sign()
    sign_output = torch.empty_like(y_pred_unperturbed).uniform(-1, 1).sign()

    # perturb data
    data_perturbed = data * sign_input
    y_pred_perturbed = torch.nn.functional.linear(data, delta) * sign_output

    y_pred_final = y_pred_unperturbed + y_pred_perturbed

    return y_pred_final

class AdaptiveFlatten(nn.Module):
    def __init__(self, dims_to_flatten=3):
        super().__init__()
        self.dims = 3
    
    def forward(self, data):
        return torch.flatten(data, start_dim=-self.dims)
            
            