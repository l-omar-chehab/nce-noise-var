"""Generative Model is a 2D Gaussian.
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


def get_model(n_comp, n_samples, seed, data_val=-0.35):
    """Gaussian Model where the tunable parameter is the covariance off-diagonal.
    """
    assert n_comp == 2, "2-dimensional Gaussian model should have n_comp equal to 2."

    if data_val == 'default':
        data_val = -0.35

    # Generative model : source space
    set_torch_seed(seed)

    # parameter
    corr = torch.tensor(data_val)   # will be made nn.Parameter later in the script

    # prior marginal
    # remove default nn.Parameter (none here) and replace it with our own
    source_mean = torch.zeros(n_comp)
    source_cov = torch.Tensor([[1, data_val], [data_val, 1]])
    prior_marginal = MultivariateNormal(
        loc=source_mean,
        covariance_matrix=source_cov
    )

    # source
    source = prior_marginal.sample(sample_shape=(n_samples,))

    # initialize learnable parameter (at optimum)
    prior_marginal.cov_offdiag = nn.Parameter(corr, requires_grad=True)

    # Generative model : forward (x = As) and inverse (s = Wx) operators
    # (torch automatically transposes to x.T = s.T A.T)

    # forward operator
    A = torch.eye(n_comp)
    mixing_net = nn.Linear(n_comp, n_comp, bias=False)
    del mixing_net.weight
    mixing_net.weight = torch.Tensor(n_comp, n_comp)
    mixing_net.weight.data = A

    # inverse operator
    encoder = nn.Linear(n_comp, n_comp, bias=False)
    encoder.weight.data = A.T
    # Replace parameters with tensors (vs. freeze, with keeps Params with disables their grads)
    params, names = extract_weights(encoder, as_param=False)
    delete_weights(encoder)
    load_weights(encoder, names, params)

    # build flow
    data_marginal = BaseFlow(
        prior_marginal=prior_marginal,
        embedder=encoder,
        generator_true=mixing_net
    )

    return data_marginal, source
