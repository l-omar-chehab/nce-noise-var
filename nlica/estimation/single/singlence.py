"""Noise-Contrastive Estimation (NCE) over single samples: is x data or noise?"""

import torch
from torch import nn
import time
import numpy as np

from nlica.utils import set_torch_seed
from nlica.dataset import SimpleDataset


supervised = True


def make_dataset(
    obs=None,
    noise_marginal=None,
    noise_to_data=1,
    seed=42
):
    set_torch_seed(seed)

    n_data = len(obs)

    # noise
    n_noise = int(noise_to_data * n_data)
    noise = noise_marginal.sample((n_noise,))
    y_noise = torch.zeros(n_noise)
    y_obs = torch.ones(n_data)

    # formatting them in train and test datasets and dataloaders
    X = torch.cat([obs, noise], axis=0)
    y = torch.cat([y_obs, y_noise], axis=0)

    # print("obs: ", obs.shape)
    # print("noise: ", noise.shape)
    # print("X: ", X.shape)

    dataset = SimpleDataset(X=X, y=y)

    return dataset


class Model(nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        n_comp=2,
        data_marginal_truth=None,
        data_marginal=None,
        noise_marginal=None,
        noise_to_data=1
    ):
        super().__init__()
        self.n_comp = n_comp
        self.data_marginal_truth = data_marginal_truth
        self.data_marginal = data_marginal
        self.noise_marginal = noise_marginal
        self.noise_to_data = torch.tensor(noise_to_data)
        self.prop_noise = noise_to_data / (1 + noise_to_data)

    def forward(self, x):
        """Forward Pass of the network. Discriminator.
        Input:
            x (torch Tensor): of size (batch_size, n_components)
        """
        return self._discriminator(x)

    def score(self, x):
        score = self.data_marginal_truth.score(x)
        return score

    def predict_mse(self, x, n_samples, verbose=False):
        """Vectorized."""
        n_samples_montecarlo = x.shape[0]
        n_samples_data = int((1 - self.prop_noise) * n_samples)

        start = time.time()
        scores = self.score(x)               # (n_samples, n_comp_all_params)
        end = time.time()
        duration = np.round(end - start, 2)
        if verbose:
            print(f"Computing the score takes {duration} seconds")

        start = time.time()
        posteriors = self._discriminator_optimal(x)        # (n_samples,)
        posteriors = posteriors.reshape(-1, 1)   # (n_samples, 1)
        posteriors_neg = 1 - posteriors
        end = time.time()
        duration = np.round(end - start, 2)
        if verbose:
            print(f"Computing the optimal discriminator takes {duration} seconds")

        score_weighted_mean = (scores * posteriors_neg).mean(dim=0)  # (n_comp_all_params,)
        score_weighted_mean = score_weighted_mean.reshape(-1, 1)   # (n_comp_all_params, 1)
        score_weighted_covar = scores.T @ (scores * posteriors_neg)  # (n_comp_all_params, n_comp_all_params)
        score_weighted_covar = score_weighted_covar / n_samples_montecarlo
        # n_params = len(score_weighted_covar)
        # diagonal_eps = torch.diag(torch.abs(score_weighted_covar).min() * 1e-4 * torch.ones(n_params))
        diagonal_eps = torch.zeros_like(score_weighted_covar)
        score_weighted_covar = score_weighted_covar + diagonal_eps  # numerical trick for diagonalization
        # fisher_weighted_inv = torch.pinverse(score_weighted_covar)
        fisher_weighted_inv = torch.inverse(score_weighted_covar)

        asymp_cov = fisher_weighted_inv - \
            (1 + (1 / self.noise_to_data)) * \
            fisher_weighted_inv @ score_weighted_mean @ score_weighted_mean.T @ fisher_weighted_inv
        asymp_mse = torch.trace(asymp_cov) / n_samples_data

        return asymp_mse

    def _discriminator(self, x):
        """Computes parametric discriminator p(y=1 | x).

        Args:
            x (tuple of torch.Tensor): timeseries.
                                        Has shape (n_samples, n_comp).
        Returns:
            posterior (torch.Tensor): 1D tensor of probabilities.
                                        Has shape (n_samples,)
        """
        # device = next(self.parameters()).device
        # batch_size = x.shape[0]
        log_data = self.data_marginal.log_prob(x)
        log_noise = self.noise_marginal.log_prob(x)

        # logit as log-ratio
        # https://en.wikipedia.org/wiki/Logistic_regression
        logit = log_data - log_noise - torch.log(self.noise_to_data)
        y_pred = torch.sigmoid(logit)

        return y_pred

    def _discriminator_optimal(self, x):
        """Computes perfect discriminator p(y=1 | x).

        Args:
            x (tuple of torch.Tensor): timeseries.
                                        Has shape (n_samples, n_comp).
        Returns:
            posterior (torch.Tensor): 1D tensor of probabilities.
                                       Has shape (n_samples,)
        """
        # Compute posterior
        log_model = self.data_marginal_truth.log_prob(x)
        log_noise = self.noise_marginal.log_prob(x)
        weighted_logit = log_model - log_noise - torch.log(self.noise_to_data)

        posterior = torch.sigmoid(weighted_logit)

        return posterior
