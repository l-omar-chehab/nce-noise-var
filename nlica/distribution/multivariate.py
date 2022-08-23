"""Latent variable models defined by a forward operator (e.g. ICA).
"""

from nlica.distribution.base import BaseMultivariate

import itertools
from collections.abc import Iterable
import numpy as np

import torch
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
from torch.distributions import Categorical as TorchCategorical
from torch.distributions import Laplace as TorchLaplace
import torch.nn as nn


class MultivariateNormal(BaseMultivariate):
    """Multivariate Normal distribution.
    """
    def __init__(self,
                 loc=torch.zeros(1), covariance_matrix=torch.eye(1),
                 mean=None, cov_diag=None, cov_offdiag=None):
        super().__init__()
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.n_comp = len(loc)
        self.cov_diag = cov_diag
        self.cov_offdiag = cov_offdiag
        self.mean = mean

    def sample(self, sample_shape=torch.Size([])):
        normal = TorchMultivariateNormal(loc=self.get_loc(), covariance_matrix=self.get_covariance_matrix())
        samples = normal.sample(sample_shape)
        return samples

    def log_prob(self, x):
        normal = TorchMultivariateNormal(loc=self.get_loc(), covariance_matrix=self.get_covariance_matrix())
        logprob = normal.log_prob(x)
        return logprob

    def get_loc(self):
        loc = self.loc
        if self.mean is not None:
            loc = self.mean * torch.ones(self.n_comp)
        return loc

    def get_covariance_matrix(self):
        """Refreshes variables that are dependent on a parameter,
        when parameters are overwrit."""
        covariance_matrix = self.covariance_matrix
        if (self.cov_diag is not None) & (self.cov_offdiag is None):
            assert len(self.cov_diag.shape) == 0, "cov_diag is not a 0D tensor"
            covariance_matrix = self.cov_diag * torch.eye(self.n_comp)
        if (self.cov_diag is None) & (self.cov_offdiag is not None):
            assert len(self.cov_offdiag.shape) == 0, "cov_offdiag is not a 0D tensor"
            covariance_matrix = self.cov_offdiag * torch.ones((self.n_comp, self.n_comp)) - \
                self.cov_offdiag * torch.eye(self.n_comp) + \
                torch.eye(self.n_comp)
        if (self.cov_diag is not None) & (self.cov_offdiag is not None):
            assert len(self.cov_diag.shape) == 0, "cov_diag is not a 0D tensor"
            assert len(self.cov_offdiag.shape) == 0, "cov_offdiag is not a 0D tensor"
            covariance_matrix = self.cov_offdiag * torch.ones((self.n_comp, self.n_comp)) - \
                self.cov_offdiag * torch.eye(self.n_comp) + \
                self.cov_diag * torch.eye(self.n_comp)
        return covariance_matrix

    # def _log_prob_analytical(self, x):
    #     """x is (n_samples, n_comp)"""
    #     # Anticipate two cases : x is (n_samples, n_comp) or (n_comp,)
    #     is_flat = (len(x.shape) == 1)
    #     if is_flat:
    #         x = x.reshape(1, -1)

    #     precision = torch.inverse(self.cov)
    #     logprob = torch.tensor(0.)
    #     logprob += torch.log(self.cov.det().abs())
    #     logprob += torch.log(torch.tensor(2 * math.pi)**self.n_comp)
    #     logprob = logprob + (x - self.mean) @ precision.T @ (x - self.mean).T  # notinline so it broadcasts
    #     logprob *= -(1. / 2)

    #     # Anticipate two cases : x is (n_samples, n_comp) or (n_comp,)
    #     if is_flat:
    #         return logprob[0]
    #     else:
    #         return logprob

    #     return logprob

    def _score_analytical(self, x):
        """Computes the score vector of a univariate Normal distribution
        parameterized by its mean and variance.

        Inputs:
            x (torch.tensor): of shape (n_samples,) or (n_samples, n_comp)

        Refs:
        https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density
        https://stats.stackexchange.com/questions/268458/score-function-of-bivariate-multivariate-normal-distribution
        """
        # Anticipate two cases : x is (n_samples, n_comp) or (n_comp,)
        is_flat = (len(x.shape) == 1)
        if is_flat:
            x = x.reshape(1, -1)

        precision = torch.inverse(self.get_covariance_matrix())
        scores_mean, scores_cov = [], []
        for _ in x:
            score_mean = precision @ (_ - self.loc)
            temp = precision @ torch.outer(score_mean, score_mean) @ precision
            score_cov = (1. / 2) * (-precision + temp)
            # track
            scores_mean.append(score_mean)
            scores_cov.append(score_cov)
        scores_mean = torch.stack(scores_mean, dim=0)
        scores_cov = torch.stack(scores_cov, dim=0)
        # flatten
        scores_mean = scores_mean.flatten(start_dim=1)
        scores_cov = scores_cov.flatten(start_dim=1)
        # concatenate
        scores = torch.cat([scores_mean, scores_cov], dim=-1)  # (n_samples, n_params)

        # Anticipate two cases : x is (n_samples, n_comp) or (n_comp,)
        if is_flat:
            return scores[0]
        else:
            return scores


class MultivariateLaplace(BaseMultivariate):
    """Factorial Laplace distribution.

    Concatenates (independent) Laplace distributions per component.
    Each distribution is a native Pytorch distribution from torch.distributions.
    """
    def __init__(self, n_comp=2):
        super().__init__()
        self.n_comp = n_comp
        self.locs = torch.zeros(n_comp)
        self.scales = nn.Parameter(torch.ones(n_comp), requires_grad=True)

    def sample(self, sample_shape=torch.Size([])):
        # Define distributions per component
        distributions = [
            TorchLaplace(loc=loc, scale=scale) for (loc, scale) in zip(self.locs, self.scales)
        ]

        # Stack their samples
        samples = torch.stack(
            [distr.sample(sample_shape) for distr in distributions], axis=-1
        )
        return samples

    def log_prob(self, x):
        # Define distributions per component
        distributions = [
            TorchLaplace(loc=loc, scale=scale) for (loc, scale) in zip(self.locs, self.scales)
        ]

        # Stack their log-probabilities
        logprob = torch.stack(
            [
                distr.log_prob(x[..., comp])
                for distr, comp in zip(distributions, range(self.n_comp))
            ],
            axis=-1,
        )
        return logprob.sum(-1)


class Histogram(BaseMultivariate):
    """Trainable (discrete approximation of a) generic distribution over a given support.
    A histogram that is tunable with autograd.
    """
    def __init__(self, low=-5, high=5, n_bins=9, n_comp=2):
        super().__init__()
        self.n_comp = n_comp

        # bins
        self.n_bins = n_bins
        self.n_intervals = n_bins**n_comp

        # Support of the histogram
        # low is a list of lower bounds for each dim (e.g. x, y, z)
        # high is a list of upper bounds for each dim (e.g. x, y, z)
        if not isinstance(low, Iterable):
            low = [low] * n_comp
        if not isinstance(high, Iterable):
            high = [high] * n_comp
        assert len(low) == n_comp, "Length mismatch between low and n_comp."
        assert len(high) == n_comp, "Length mismatch between high and n_comp."

        # build mesh
        # each element is a list of x-coords, y-coords, etc.
        coords = [torch.linspace(lb, ub, self.n_bins + 1)
                  for (lb, ub) in zip(low, high)]
        intervals = []
        for comp in range(n_comp):
            intervals_1d = [(coords[comp][idx], coords[comp][idx + 1])
                            for idx in range(len(coords[comp]) - 1)]
            intervals.append(intervals_1d)
        self.mesh = torch.Tensor(list(itertools.product(*intervals)))
        # (n_bins, n_comp, 2), last dim for low and high vertex along a given dim

        # define pmf over the mesh
        self.masses = nn.Parameter(torch.ones(self.n_intervals, requires_grad=True))  # initialize with maxent
        self.softmax = nn.Softmax(dim=0)

    def sample(self, sample_shape=torch.Size([])):
        # make iterable
        if not isinstance(sample_shape, Iterable):
            sample_shape = (sample_shape,)
        sample_size = np.prod(sample_shape)

        # choose the bin index
        cat = TorchCategorical(probs=self.softmax(self.masses))
        bins_idx = cat.sample(sample_shape=(sample_size,))

        # sample from histogram
        samples = []
        for idx in bins_idx:
            interval = self.mesh[idx]
            low, high = interval[:, 0], interval[:, 1]
            sample = self._sample_uniform(1, self.n_comp, low, high)
            samples.append(sample)
        samples = torch.cat(samples, dim=0)  # (n_data_flat, n_comp)

        return samples.reshape(*sample_shape, self.n_comp)

    def _prob(self, x):
        '''Here x is a single point with shape (n_comp,).'''
        low, high = self.mesh[:, :, 0], self.mesh[:, :, 1]  # (n_bins, n_comp)
        indicators_per_interval = ((x[None, :] > low) & (x[None, :] < high)).prod(dim=-1)  # (n_bins,)
        not_in_support = indicators_per_interval.sum() == 0  # indicators are zero for all intervals
        if not_in_support:
            return torch.tensor(0., requires_grad=True)
            # return a new tensor so that samples outside support backprop a gradient (enables autograd)
            # but that is not meaningful
        else:
            indicators_per_interval = ((x[None, :] > low) & (x[None, :] < high)).prod(dim=-1)  # (n_bins,)
            interval_idx = np.where(indicators_per_interval)[0][0]
            return self.softmax(self.masses)[interval_idx]

    def _sample_uniform(self, sample_size, n_comp, low, high):
        # make iterable
        if not isinstance(low, Iterable):
            low = [low] * n_comp
        if not isinstance(high, Iterable):
            high = [high] * n_comp
        if not isinstance(sample_size, Iterable):
            sample_size = (sample_size,)
        assert len(low) == n_comp, "Length mismatch between low and n_comp."
        assert len(high) == n_comp, "Length mismatch between high and n_comp."

        # sample uniform
        samples = torch.stack([torch.Tensor(*sample_size).uniform_(small, big)
                               for (small, big) in zip(low, high)], dim=-1)
        return samples

    def log_prob(self, x):
        ps = torch.stack([self._prob(_) for _ in x])
        return torch.log(ps)
