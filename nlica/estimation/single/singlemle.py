"""Maximum-Likelihood Estimation (MLE) over single samples: what best fits x?"""

import torch
from torch import nn

from nlica.utils import set_torch_seed
from nlica.dataset import SimpleDataset


supervised = False


def make_dataset(
    obs=None,
    noise_marginal=None,
    noise_to_data=1,
    seed=42
):
    set_torch_seed(seed)

    n_data = len(obs)

    X, y = obs, torch.ones(n_data)

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
        """Forward Pass of the network. Negative log-likelihood.
        Input:
            x (torch Tensor): of size (batch_size, n_components)
        """
        # device = next(self.parameters()).device
        # batch_size = x.shape[0]
        logprob = self.data_marginal.log_prob(x)
        return -logprob

    def score(self, x):
        score = self.data_marginal_truth.score(x)
        return score

    def predict_mse(self, x, n_samples, verbose=False):
        """Vectorized."""
        n_samples_montecarlo = x.shape[0]
        n_samples_data = int((1 - self.prop_noise) * n_samples)

        scores = self.score(x)  # (n_samples, n_comp_all_params)
        score_covar = scores.T @ scores  # (n_comp_all_params, n_comp_all_params)
        score_covar = score_covar / n_samples_montecarlo
        fisher_inv = torch.inverse(score_covar)
        asymp_cov = fisher_inv
        asymp_var = torch.trace(asymp_cov) / n_samples_data
        return asymp_var
