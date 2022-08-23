"""Generative Model is a 1D Gaussian.
"""

import torch
import torch.nn as nn

from nlica.distribution.base import BaseFlow
from nlica.distribution.multivariate import MultivariateNormal
from nlica.utils import (
    extract_weights,
    load_weights,
    delete_weights,
    set_torch_seed
)


def get_model(n_comp, n_samples,
              seed, data_val=0.):
    """Gaussian Model where the tunable parameter is the mean diagonal.
    """
    assert n_comp == 1, "1-dimensional Gaussian model should have n_comp equal to 1."

    if data_val == 'default':
        data_val = 0.

    # Generative model : source space
    set_torch_seed(seed)

    # param
    mean = torch.tensor(data_val)  # will be made nn.Parameter later in the script

    # prior marginal
    # remove default nn.Parameter (none here) and replace it with our own
    source_mean = mean * torch.ones(n_comp)
    source_cov = torch.eye(n_comp)
    prior_marginal = MultivariateNormal(
        loc=source_mean,
        covariance_matrix=source_cov
    )

    # source
    source = prior_marginal.sample(sample_shape=(n_samples,))

    # initialize learnable parameter (at optimum)
    prior_marginal.mean = nn.Parameter(mean, requires_grad=True)

    # Generative model : forward (x = As) and inverse (s = Wx) operators
    # (torch automatically transposes to x.T = s.T A.T)

    # forward operator
    A = torch.eye(n_comp)
    mixing_net = nn.Linear(n_comp, n_comp, bias=False)
    del mixing_net.weight
    mixing_net.weight = torch.Tensor(n_comp, n_comp)
    mixing_net.weight.data = A

    # inverse operator
    embedder = nn.Linear(n_comp, n_comp, bias=False)
    embedder.weight.data = A.T
    # Replace parameters with tensors (vs. freeze, which keeps Params but disables their grads)
    params, names = extract_weights(embedder, as_param=False)
    delete_weights(embedder)
    load_weights(embedder, names, params)

    # build flow
    data_marginal = BaseFlow(
        prior_marginal=prior_marginal,
        embedder=embedder,
        generator_true=mixing_net
    )

    return data_marginal, source
