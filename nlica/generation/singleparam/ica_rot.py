"""Generative Model is a 2D ICA model, where the sources are AR with Laplace innovations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlica.distribution.base import BaseFlow
from nlica.distribution.multivariate import MultivariateLaplace
from nlica.utils import set_torch_seed


class Rotation(nn.Linear):
    def __init__(self, angle):
        super(Rotation, self).__init__(in_features=2, out_features=2, bias=False)
        self.angle = angle
        del self.weight

    def forward(self, x):
        x_rotated = F.linear(x, self.get_weight(), self.bias)
        return x_rotated

    def get_weight(self):
        """Update the rotation matrix if the angle is overwritten."""
        weight = torch.zeros(2, 2)
        weight[0, 0] = torch.cos(self.angle)
        weight[0, 1] = torch.sin(self.angle)
        weight[1, 0] = -torch.sin(self.angle)
        weight[1, 1] = torch.cos(self.angle)
        return weight


def get_model(n_comp, n_samples, seed, data_val=math.pi / 4):
    """ICA Model where the tunable parameter is the encoder (angle).
    """
    assert n_comp == 2, "2-dimensional Orthogonal ICA model should have n_comp equal to 2."

    if data_val == 'default':
        data_val = math.pi / 4

    # Generative model : source space
    set_torch_seed(seed)

    # param
    angle = torch.tensor(data_val)  # will be made nn.Parameter later in the script

    # prior marginal
    # remove default nn.Parameter
    prior_marginal = MultivariateLaplace(n_comp=n_comp)
    del prior_marginal.scales
    prior_marginal.scales = torch.linspace(1, 20, n_comp)

    # source
    source = prior_marginal.sample(sample_shape=(n_samples,))

    # Generative model : forward (x = As) and inverse (s = Wx) operators
    # (torch automatically transposes to x.T = s.T A.T)

    # forward
    # for orthogonal ICA, the forward operator is a rotation matrix
    # A = torch.Tensor(gen_ortho(dim=n_comp, seed=seed_mixing))
    mixing_net = Rotation(angle=angle)

    # inverse
    embedder = Rotation(angle=nn.Parameter(-angle, requires_grad=True))

    # build flow
    data_marginal = BaseFlow(
        prior_marginal=prior_marginal,
        embedder=embedder,
        generator_true=mixing_net
    )

    return data_marginal, source
